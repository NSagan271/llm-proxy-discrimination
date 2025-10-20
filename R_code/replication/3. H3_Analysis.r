
# setup -----------------------------------------------------------------

library(dplyr)
library(tidyr)
library(stringr)
library(purrr)
library(broom)
library(lme4)
library(fixest)
library(kableExtra)
library(knitr)

set.seed(42)
fixest::setFixest_notes(FALSE)

dir.create("Tables/H3", recursive = TRUE, showWarnings = FALSE)

human_trials <- readRDS("data_derived/human_scored.rds")  # trial-level human data
llm_raw      <- readRDS("data_derived/llm_raw.rds")       # profile×model rows, with certainty_score


# helpers -----------------------------------------------------------------

fmt_or <- function(OR, lo, hi, digits = 2) {
  sprintf(paste0("%.", digits, "f (%.", digits, "f, %.", digits, "f)"), OR, lo, hi)
}
fmt_p <- function(x, digits = 3, lt_threshold = 10^-digits) {
  ifelse(
    is.na(x), "",
    ifelse(x < lt_threshold,
           paste0("<", formatC(lt_threshold, format = "f", digits = digits)),
           formatC(x, format = "f", digits = digits))
  )
}

# Write a compact longtable to LaTeX (no escaping so math is OK)
write_longtable <- function(df, file, caption, align = NULL) {
  if (is.null(align)) align <- rep("l", ncol(df))
  kbl(df,
      format    = "latex",
      booktabs  = TRUE,
      longtable = TRUE,
      align     = align,
      caption   = caption,
      escape    = FALSE
  ) |>
    kable_styling(latex_options = "repeat_header") |>
    save_kable(file)
}


# H3a/H3b (humans) --------------------------------------------------------

# Map attribute -> outcome column in human_trials
human_attr_map <- list(
  "Sex"                   = "sex_correct_overall",
  "Income"                = "income_correct_overall",
  "Age \\(\\ge 40\\)"     = "agebin_correct_overall",
  "Age \\(\\pm 5\\) years"= "agein5_correct_overall"  # appendix metric
)

fit_h3_humans <- function(dat, treatment_value, attr_label) {
  ycol <- human_attr_map[[attr_label]]
  dd <- dat |>
    filter(treatment == treatment_value,
           !is.na(certainty),
           !is.na(.data[[ycol]])) |>
    mutate(
      certainty  = as.numeric(certainty),           # 1..5 step = one-unit change
      profile_id = factor(profile_id),
      ResponseId = factor(ResponseId)
    )
  
  if (nrow(dd) == 0) {
    return(tibble(
      Attribute = attr_label, OR = NA_real_, OR_lo = NA_real_, OR_hi = NA_real_,
      p = NA_real_, N = 0, N_profiles = 0, N_respondents = 0
    ))
  }
  
  # Mixed-effects logistic: correctness ~ certainty + (1|profile) + (1|respondent)
  fit <- suppressWarnings(
    glmer(
      formula = as.formula(paste0(ycol, " ~ certainty + (1|profile_id) + (1|ResponseId)")),
      data    = dd,
      family  = binomial(),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
  )
  
  s <- summary(fit)$coefficients
  if (!"certainty" %in% rownames(s)) {
    return(tibble(
      Attribute = attr_label, OR = NA_real_, OR_lo = NA_real_, OR_hi = NA_real_,
      p = NA_real_, N = nrow(dd), N_profiles = n_distinct(dd$profile_id),
      N_respondents = n_distinct(dd$ResponseId)
    ))
  }
  
  est <- s["certainty", "Estimate"]
  se  <- s["certainty", "Std. Error"]
  z   <- s["certainty", "z value"]
  p   <- 2 * pnorm(abs(z), lower.tail = FALSE)
  
  OR    <- exp(est)
  OR_lo <- exp(est - 1.96 * se)
  OR_hi <- exp(est + 1.96 * se)
  
  tibble(
    Attribute = attr_label,
    OR = OR, OR_lo = OR_lo, OR_hi = OR_hi, p = p,
    N = nrow(dd),
    N_profiles = n_distinct(dd$profile_id),
    N_respondents = n_distinct(dd$ResponseId)
  )
}

# Fit and write H3a (humans, no timer) and H3b (humans, time pressure)
do_h3_humans <- function(dat, treatment_value, outfile, caption_prefix, console_tag) {
  res <- bind_rows(
    fit_h3_humans(dat, treatment_value, "Sex"),
    fit_h3_humans(dat, treatment_value, "Income"),
    fit_h3_humans(dat, treatment_value, "Age \\(\\ge 40\\)"),
    fit_h3_humans(dat, treatment_value, "Age \\(\\pm 5\\) years")
  ) |>
    mutate(
      `OR per +1 certainty (95\\% CI)` = fmt_or(OR, OR_lo, OR_hi),
      `p-value` = fmt_p(p, 3)
    ) |>
    select(
      Attribute,
      `OR per +1 certainty (95\\% CI)`,
      `p-value`,
      N, `Profiles (N)` = N_profiles, `Respondents (N)` = N_respondents
    )
  
  write_longtable(
    res,
    file = outfile,
    caption = paste0(
      caption_prefix,
      "Mixed-effects logistic regression with random intercepts for profile and respondent; ",
      "certainty treated as a 1–5 ordered predictor. ORs reflect a one-step increase in certainty."
    ),
    align = c("l","l","r","r","r","r")
  )
  
  # Console summary for main text
  cat("\n", console_tag, "\n", sep = "")
  print(res, n = 50)
  invisible(res)
}

h3a_tbl <- do_h3_humans(
  human_trials, "control",
  outfile = "Tables/H3/h3a_humans_notimer.tex",
  caption_prefix = "H3a (Humans, no timer): certainty–accuracy association. ",
  console_tag = "H3a (Humans, no timer):"
)

h3b_tbl <- do_h3_humans(
  human_trials, "timepressure",
  outfile = "Tables/H3/h3b_humans_timepressure.tex",
  caption_prefix = "H3b (Humans, time pressure): certainty–accuracy association. ",
  console_tag = "H3b (Humans, time pressure):"
)


# H3c (LLMs) --------------------------------------------------------------

# Attribute -> outcome column in llm_raw
llm_attr_map <- list(
  "Sex"                   = "successes_sex_overall",
  "Income"                = "successes_income_overall",
  "Age \\(\\ge 40\\)"     = "successes_agebin_overall",
  "Age \\(\\pm 5\\) years"= "successes_agein5_overall"
)

# Extract per-model certainty slope (OR per +1 certainty) from a fixest feglm
get_per_model_slopes <- function(fit, model_levels) {
  coefs <- broom::tidy(fit)
  base <- coefs |>
    filter(term == "certainty_score") |>
    transmute(base_est = estimate, base_se = std.error)
  if (nrow(base) == 0) {
    stop("Base certainty_score term not found; check model or factor coding.")
  }
  base_est <- base$base_est[1]; base_se <- base$base_se[1]
  
  inter <- coefs |>
    filter(str_starts(term, "certainty_score:model")) |>
    mutate(level = sub("^certainty_score:model", "", term)) |>
    select(level, est = estimate, se = std.error)
  
  purrr::map_dfr(model_levels, function(m) {
    int_row <- inter |> filter(level == m)
    if (nrow(int_row) == 0) {
      est <- base_est
      se  <- base_se
    } else {
      est <- base_est + int_row$est
      # Conservative SE ignoring covariance
      se  <- sqrt(base_se^2 + int_row$se^2)
    }
    z <- est / se
    p <- 2 * pnorm(abs(z), lower.tail = FALSE)
    tibble(model = m,
           est = est, se = se, z = z, p = p,
           OR = exp(est),
           OR_lo = exp(est - 1.96*se),
           OR_hi = exp(est + 1.96*se))
  })
}

# Bootstrap per-model certainty slopes by resampling profiles (returns 95% CIs)
bootstrap_slopes <- function(dat, B = 5000, quiet = TRUE) {
  profs <- unique(dat$profile)
  model_levels <- levels(dat$model)
  
  base_fit <- suppressWarnings(
    fixest::feglm(
      correct ~ certainty_score * model | profile,
      family = binomial(),
      weights = ~ n,
      data = dat
    )
  )
  base_slopes <- get_per_model_slopes(base_fit, model_levels)
  
  draws <- vector("list", B)
  for (b in seq_len(B)) {
    samp <- sample(profs, length(profs), replace = TRUE)
    tab  <- as.integer(table(factor(samp, levels = profs)))
    d_b  <- dat |>
      mutate(boot_w = tab[match(profile, profs)]) |>
      filter(boot_w > 0)
    
    est_b <- try({
      fit_b <- suppressWarnings(
        fixest::feglm(
          correct ~ certainty_score * model | profile,
          family  = binomial(),
          weights = ~ n * boot_w,     # <-- keep multiplicity
          data    = d_b
        )
      )
      get_per_model_slopes(fit_b, model_levels)
    }, silent = TRUE)
    
    if (!inherits(est_b, "try-error")) draws[[b]] <- est_b
    if (!quiet && b %% 250 == 0) message("Bootstrap iter: ", b)
  }
  
  draws <- purrr::compact(draws)
  if (length(draws) == 0) {
    warning("All bootstrap fits failed; returning Wald CIs only.")
    return(base_slopes |>
             transmute(model, OR_lo_boot = OR_lo, OR_hi_boot = OR_hi))
  }
  
  bind_rows(draws, .id = "iter") |>
    group_by(model) |>
    summarise(
      OR_lo_boot = quantile(OR, 0.025, na.rm = TRUE),
      OR_hi_boot = quantile(OR, 0.975, na.rm = TRUE),
      .groups = "drop"
    )
}

fit_h3c_one_attr <- function(dat, attr_label, B = 5000, quiet_boot = TRUE) {
  ycol <- llm_attr_map[[attr_label]]
  dd <- dat |>
    filter(!is.na(.data[[ycol]]),
           !is.na(certainty_score)) |>
    transmute(
      profile = factor(profile),
      model   = factor(arm),                 # per-model interactions
      certainty_score = as.numeric(certainty_score),
      correct = as.numeric(.data[[ycol]] / pmax(trials_overall, 1L)),
      n       = pmax(trials_overall, 1L)
    )
  
  fit <- suppressWarnings(
    fixest::feglm(
      correct ~ certainty_score * model | profile,
      family = binomial(),
      weights = ~ n,
      data = dd
    )
  )
  
  model_levels <- levels(dd$model)
  slopes <- get_per_model_slopes(fit, model_levels) |>
    mutate(p_BH = p.adjust(p, method = "BH"))
  
  boot_ci <- bootstrap_slopes(dd, B = B, quiet = quiet_boot)
  
  res <- slopes |>
    left_join(boot_ci, by = "model") |>
    mutate(
      OR_lo_use = ifelse(is.na(OR_lo_boot), OR_lo, OR_lo_boot),
      OR_hi_use = ifelse(is.na(OR_hi_boot), OR_hi, OR_hi_boot),
      Attribute = attr_label,
      `OR per +1 certainty (95\\% CI)` = fmt_or(OR, OR_lo_use, OR_hi_use),
      `p-value`  = fmt_p(p, 3),
      `q-value (BH)` = fmt_p(p_BH, 3)
    ) |>
    arrange(desc(OR)) |>
    select(Attribute, Arm = model,
           `OR per +1 certainty (95\\% CI)`, `p-value`, `q-value (BH)`)
  
  list(table = res,
       N_profiles = n_distinct(dd$profile),
       N_models   = n_distinct(dd$model))
}

# Fit all four attributes, then bind into ONE H3c longtable
h3c_sex    <- fit_h3c_one_attr(llm_raw, "Sex",                 B = 1000, quiet_boot = TRUE)
h3c_income <- fit_h3c_one_attr(llm_raw, "Income",              B = 1000, quiet_boot = TRUE)
h3c_age40  <- fit_h3c_one_attr(llm_raw, "Age \\(\\ge 40\\)",   B = 1000, quiet_boot = TRUE)
h3c_age5   <- fit_h3c_one_attr(llm_raw, "Age \\(\\pm 5\\) years", B = 1000, quiet_boot = TRUE)

h3c_all_tab <- bind_rows(h3c_sex$table, h3c_income$table, h3c_age40$table, h3c_age5$table)

write_longtable(
  h3c_all_tab,
  file = "Tables/H3/h3c_llms_allattrs.tex",
  caption = paste0(
    "H3c (LLMs): certainty–accuracy association (all attributes). ",
    "Profile fixed-effects binomial GLM with certainty \\(\\times\\) model interactions; ",
    "ORs are per one-step increase in certainty. 95\\% CIs via bootstrap over profiles (B = 1{,}000); ",
    "Benjamini--Hochberg correction applied within attribute."
  ),
  align = c("l","l","l","r","r")
)


# summaries ---------------------------------------------------------------

summarise_h3c <- function(res_list, tag) {
  df <- res_list$table
  or_num <- as.numeric(sub("^([0-9.]+).*", "\\1", df$`OR per +1 certainty (95\\% CI)`))
  q_str <- df$`q-value (BH)`
  q_num <- suppressWarnings(as.numeric(sub("^<", "", q_str)))
  sig   <- !is.na(q_num) & q_num < 0.05
  tibble(
    attribute = unique(df$Attribute),
    models_total = nrow(df),
    median_OR = median(or_num, na.rm = TRUE),
    min_OR = min(or_num, na.rm = TRUE),
    max_OR = max(or_num, na.rm = TRUE),
    sig_BH = sum(sig, na.rm = TRUE)
  ) |>
    mutate(tag = tag) |>
    select(tag, everything())
}

cat("\nH3c (LLMs) — per-attribute summary (report in main text):\n")
print(bind_rows(
  summarise_h3c(h3c_sex,   "Sex"),
  summarise_h3c(h3c_income,"Income"),
  summarise_h3c(h3c_age40, "Age >=40"),
  summarise_h3c(h3c_age5,  "Age ±5y")
), n = 50)

cat("\nTop/bottom 5 models by certainty slope OR (per attribute):\n")
show_top_bottom <- function(res_list, k = 5) {
  df <- res_list$table
  or_num <- as.numeric(sub("^([0-9.]+).*", "\\1", df$`OR per +1 certainty (95\\% CI)`))
  df2 <- df |> mutate(OR_num = or_num)
  cat("\n", unique(df2$Attribute), "\n", sep = "")
  print(df2 |> arrange(desc(OR_num)) |> head(k) |> select(Arm, `OR per +1 certainty (95\\% CI)`, `q-value (BH)`))
  print(df2 |> arrange(OR_num) |> head(k)       |> select(Arm, `OR per +1 certainty (95\\% CI)`, `q-value (BH)`))
}
show_top_bottom(h3c_sex)
show_top_bottom(h3c_income)
show_top_bottom(h3c_age40)
show_top_bottom(h3c_age5)

cat("\nDone. LaTeX outputs:\n",
    "- Tables/H3/h3a_humans_notimer.tex\n",
    "- Tables/H3/h3b_humans_timepressure.tex\n",
    "- Tables/H3/h3c_llms_allattrs.tex\n", sep = "")
