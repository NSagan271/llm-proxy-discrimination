
# setup ----------------------------------------------------------------

library(dplyr)
library(readr)
library(purrr)
library(tidyr)
library(knitr)
library(kableExtra)
library(stringr)
library(ggplot2)


set.seed(42)

combined_cert3 <- readRDS("data_derived/combined_cert3.rds")

dir.create("Figures/RQ1", recursive = TRUE, showWarnings = FALSE)
dir.create("Tables/RQ1",  recursive = TRUE, showWarnings = FALSE)

df <- combined_cert3


# helpers from H1/H2 --------------------------------------------------------------

# Bootstrap over profiles for the paired profile gap
# Returns a tibble with mean gap and 95% CI by arm
paired_profile_gap <- function(df, successes_col, trials_col, B = 5000) {
  # Keep only arms that have data
  arms <- unique(df$arm)
  
  # Per-profile accuracy by arm
  acc <- df %>%
    transmute(profile, arm,
              acc = .data[[successes_col]] / pmax(.data[[trials_col]], 1L)) %>%
    tidyr::pivot_wider(names_from = arm, values_from = acc)
  
  # Profiles usable for gap (must have human_notimer and the model arm value)
  profs <- acc$profile
  
  # Compute observed mean gap per non-human_notimer arm
  arms_to_compare <- setdiff(arms, "human_notimer")
  obs <- map_dfr(arms_to_compare, function(a) {
    # vector of per-profile gaps; remove NAs created by missing acc
    gaps <- acc[[a]] - acc[["human_notimer"]]
    tibble(arm = a,
           mean_gap = mean(gaps, na.rm = TRUE),
           n_profiles = sum(!is.na(gaps)))
  })
  
  # Bootstrap profiles with replacement
  Bmat <- replicate(B, {
    samp <- sample(profs, length(profs), replace = TRUE)
    # join sample back to acc and compute mean gaps per arm
    acc_s <- acc[match(samp, acc$profile), , drop = FALSE]
    vapply(arms_to_compare, function(a) {
      mean(acc_s[[a]] - acc_s[["human_notimer"]], na.rm = TRUE)
    }, numeric(1))
  })
  
  # Summarize bootstrap CIs
  ci <- t(apply(Bmat, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE))
  ci <- as.data.frame(ci)
  ci$arm <- arms_to_compare
  
  obs %>%
    left_join(ci %>% rename(ci_lo = `2.5%`, ci_hi = `97.5%`), by = "arm") %>%
    select(arm, n_profiles, mean_gap, ci_lo, ci_hi) %>%
    arrange(desc(mean_gap))
}

# Fit fixest grouped-binomial FE GLM for one attribute and return tidy table
fit_h1_one <- function(df, successes_col, trials_col, B = 5000) {
  dd <- df %>%
    mutate(
      arm = stats::relevel(factor(arm), ref = "human_notimer"),
      y1  = .data[[successes_col]],
      n   = .data[[trials_col]],
      p   = ifelse(n > 0, y1 / n, NA_real_)
    )
  
  # Base FE-GLM for point estimates / p-values
  fit <- fixest::feglm(
    p ~ arm | profile,
    family  = binomial(),
    weights = ~ n,
    data    = dd
  )
  
  # Tidy base fit
  summ <- broom::tidy(fit) %>%
    dplyr::filter(term != "(Intercept)") %>%
    dplyr::mutate(arm = sub("^arm", "", term),
                  OR  = exp(estimate))
  
  # bootstrap over profiles (CIs)
  profs <- unique(dd$profile)
  arm_terms <- summ$term
  
  # Resample profiles with replacement and refit
  boot_mat <- replicate(B, {
    # sample profiles with replacement
    samp <- sample(profs, length(profs), replace = TRUE)
    
    # turn multiplicities into a weight per profile
    freq <- as.integer(table(factor(samp, levels = profs)))
    dsub <- dd %>%
      dplyr::mutate(boot_w = freq[match(profile, profs)]) %>%
      dplyr::filter(boot_w > 0)
    
    f <- tryCatch(
      fixest::feglm(
        p ~ arm | profile,
        family  = binomial(),
        weights = ~ n * boot_w,   # <<--- key: apply bootstrap multiplicity
        data    = dsub
      ),
      error = function(e) NULL
    )
    
    if (is.null(f)) {
      out <- rep(NA_real_, length(arm_terms)); names(out) <- arm_terms; out
    } else {
      co <- stats::coef(f)
      out <- rep(NA_real_, length(arm_terms)); names(out) <- arm_terms
      out[names(co)] <- co
      out
    }
  })
  
  rownames(boot_mat) <- arm_terms
  
  # Convert bootstraps to percentile CIs and attach to table
  boot_ci <- apply(boot_mat, 1, stats::quantile, probs = c(0.025, 0.975), na.rm = TRUE) %>%
    t() %>%
    as.data.frame() %>%
    tibble::rownames_to_column("term") %>%
    dplyr::rename(ci_lo = `2.5%`, ci_hi = `97.5%`)
  
  summ <- summ %>%
    dplyr::left_join(boot_ci, by = c("term" = "term")) %>%
    dplyr::mutate(OR_lo = exp(ci_lo), OR_hi = exp(ci_hi))
  
  # BH only for non-human_timepressure arms
  summ$p_BH <- NA_real_
  idx <- which(summ$arm != "human_timepressure")
  if (length(idx) > 0) summ$p_BH[idx] <- p.adjust(summ$p.value[idx], method = "BH")
  
  summ <- summ %>% dplyr::select(arm, estimate, std.error, statistic, p.value, p_BH, OR, OR_lo, OR_hi)
  
  list(fit = fit, table = summ)
}

sanitize_latex <- function(x) {
  x %>%
    stringr::str_replace_all("×", "\\\\(\\\\times\\\\)") %>%  # 8×22B -> 8 \( \times \) 22B
    stringr::str_replace_all("—", "---") %>%                  # em dash -> LaTeX em-dash
    stringr::str_replace_all("–", "--")                       # en dash -> LaTeX en-dash
}

extract_or_num <- function(or_str) {
  as.numeric(stringr::str_match(or_str, "^([0-9]+\\.?[0-9]*)")[,2])
}

add_baseline <- function(gap_tbl) {
  tibble(
    arm        = "human_notimer",
    n_profiles = max(gap_tbl$n_profiles, na.rm = TRUE),
    mean_gap   = 0,
    ci_lo      = 0,
    ci_hi      = 0
  ) %>% bind_rows(gap_tbl)
}

# Humans at top, then LLMs by mean_pp (desc) within each figure
order_one_facet <- function(df_fac) {
  df_fac <- df_fac %>%
    mutate(
      order_index = dplyr::case_when(
        arm == "human_notimer"      ~ 1L,
        arm == "human_timepressure" ~ 2L,
        TRUE                        ~ 2L + as.integer(rank(-mean_pp, ties.method = "first"))
      )
    )
  lev <- df_fac %>%
    arrange(order_index) %>%
    pull(arm_label) %>%
    unique() %>%
    rev()                 # reverse so humans plot at the top of y-axis
  df_fac %>% mutate(y_fac = factor(arm_label, levels = lev))
}

# single-attribute plotting helper
make_gap_plot <- function(gap_tbl, file_stub) {
  df <- add_baseline(gap_tbl) %>%
    mutate(
      arm_label = case_when(
        arm == "human_notimer"      ~ "Human (no timer)",
        arm == "human_timepressure" ~ "Human (time pressure)",
        TRUE                        ~ arm
      ),
      mean_pp  = 100 * mean_gap,
      lo_pp    = 100 * ci_lo,
      hi_pp    = 100 * ci_hi,
      is_human = arm %in% c("human_notimer","human_timepressure")
    ) %>%
    order_one_facet()
  
  p <- ggplot(
    df,
    aes(x = mean_pp, y = y_fac,
        xmin = lo_pp, xmax = hi_pp,
        color = ifelse(is_human, arm_label, "LLM"),
        shape = ifelse(is_human, arm_label, "LLM"))
  ) +
    guides(color = guide_legend(
      override.aes = list(shape = c(15, 17, 16))  # square, triangle, circle
    )) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5, alpha = 0.6) +
    geom_pointrange(linewidth = 0.5) +
    scale_x_continuous(labels = scales::label_number(accuracy = 1)) +
    scale_color_manual(
      values = c(
        "Human (no timer)"      = "#000000",
        "Human (time pressure)" = "#1b73e8",
        "LLM"                   = "#333333"
      ),
      breaks = c("Human (no timer)", "Human (time pressure)", "LLM"),
      name   = NULL
    ) +
    scale_shape_manual(
      values = c("Human (no timer)" = 15, "Human (time pressure)" = 17, "LLM" = 16),
      breaks = c("Human (no timer)", "Human (time pressure)", "LLM"),
      guide  = "none"
    ) +
    labs(
      x = "Mean percentage-point gap vs Human (no timer)",
      y = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.position  = "bottom",
      legend.direction = "horizontal",
      plot.title       = element_blank(),
      plot.subtitle    = element_blank()
    )
  
  ggsave(sprintf("Figures/%s.png", file_stub), p, width = 8.5, height = 9, dpi = 300)
  p
}

fmt_or <- function(OR, lo, hi, digits = 2) {
  sprintf(paste0("%.", digits, "f (%.", digits, "f, %.", digits, "f)"),
          OR, lo, hi)
}

fmt_p <- function(x, digits = 3, lt_threshold = 10^-digits) {
  ifelse(
    is.na(x), "",
    ifelse(
      x < lt_threshold,
      paste0("<", formatC(lt_threshold, format = "f", digits = digits)),
      formatC(x, format = "f", digits = digits)
    )
  )
}

# figures -----------------------------------------------------------------

# Gaps (paired-by-profile) vs Human (no timer)
gap_sex_rq1  <- paired_profile_gap(df, "successes_sex_overall",    "trials_overall")
gap_age_rq1  <- paired_profile_gap(df, "successes_agebin_overall", "trials_overall")
gap_age5_rq1 <- paired_profile_gap(df, "successes_agein5_overall", "trials_overall")
gap_inc_rq1  <- paired_profile_gap(df, "successes_income_overall", "trials_overall")

# Figures (write to RQ1 subfolder)
p_sex_rq1  <- make_gap_plot(gap_sex_rq1,  "RQ1/fig_sex_gap_rq1")
p_age_rq1  <- make_gap_plot(gap_age_rq1,  "RQ1/fig_agebin_gap_rq1")
p_age5_rq1 <- make_gap_plot(gap_age5_rq1, "RQ1/fig_age5_gap_rq1")
p_inc_rq1  <- make_gap_plot(gap_inc_rq1,  "RQ1/fig_income_gap_rq1")


# tables ------------------------------------------------------------------

# GLMs with profile FE (BH on model contrasts already handled in fit_h1_one)
h1_sex_rq1  <- fit_h1_one(df, "successes_sex_overall",    "trials_overall", B = 1000)
h1_age_rq1  <- fit_h1_one(df, "successes_agebin_overall", "trials_overall", B = 1000)
h1_age5_rq1 <- fit_h1_one(df, "successes_agein5_overall", "trials_overall", B = 1000)
h1_inc_rq1  <- fit_h1_one(df, "successes_income_overall", "trials_overall", B = 1000)

agebin_label <- "Age (\\( \\ge 40 \\))"
age5_label   <- "Age (\\( \\pm 5 \\) years)"

t_sex_rq1  <- h1_sex_rq1$table  %>% dplyr::mutate(attribute = "Sex",                 .before = 1)
t_inc_rq1  <- h1_inc_rq1$table  %>% dplyr::mutate(attribute = "Income",              .before = 1)
t_age_rq1  <- h1_age_rq1$table  %>% dplyr::mutate(attribute = agebin_label,          .before = 1)
t_age5_rq1 <- h1_age5_rq1$table %>% dplyr::mutate(attribute = age5_label,            .before = 1)

rq1_tbl <- dplyr::bind_rows(t_sex_rq1, t_inc_rq1, t_age_rq1, t_age5_rq1) %>%
  dplyr::mutate(
    arm_label = dplyr::case_when(
      arm == "human_timepressure" ~ "Human (time pressure)",
      TRUE                        ~ arm
    ),
    `Odds ratio (95% CI)` = fmt_or(OR, OR_lo, OR_hi),
    `p-value`             = fmt_p(p.value, 3),
    `q-value (BH)`        = dplyr::if_else(is.na(p_BH), "", fmt_p(p_BH, 3))
  ) %>%
  dplyr::select(
    Attribute = attribute,
    Arm = arm_label,
    `Odds ratio (95% CI)`,
    `p-value`,
    `q-value (BH)`
  ) %>%
  dplyr::arrange(Attribute, factor(Arm, levels = c("Human (time pressure)")), Arm)

write_attr_longtable <- function(df, attr, file_stub,
                                 out_dir = "Tables",     # <-- new
                                 caption_prefix = "") {  # <-- new
  df_attr <- df %>%
    dplyr::filter(Attribute == attr) %>%
    dplyr::mutate(
      is_htp = Arm == "Human (time pressure)",
      OR_num = extract_or_num(`Odds ratio (95% CI)`),
      Attribute = sanitize_latex(Attribute),
      Arm       = sanitize_latex(Arm),
      `Odds ratio (95% CI)` = sanitize_latex(`Odds ratio (95% CI)`),
      `p-value`             = `p-value`,
      `q-value (BH)`        = `q-value (BH)`
    ) %>%
    dplyr::arrange(dplyr::desc(is_htp), dplyr::desc(OR_num)) %>%
    dplyr::select(-is_htp, -OR_num)
  
  cap <- paste0(
    caption_prefix,
    sanitize_latex(attr),
    " --- grouped-binomial FE GLM odds ratios (vs Human no timer). ",
    "BH q-values shown only for model contrasts; Human (time pressure) is unadjusted."
  )
  
  kbl(
    df_attr,
    format    = "latex",
    booktabs  = TRUE,
    longtable = TRUE,
    align     = c("l","l","l","r","r"),
    caption   = cap,
    escape    = FALSE,
    col.names = c("Attribute",
                  "Arm",
                  "Odds ratio (95\\% CI)",
                  "p-value",
                  "q-value (BH)")
  ) %>%
    kable_styling(latex_options = "repeat_header") %>%
    save_kable(file.path(out_dir, paste0(file_stub, ".tex")))
}

write_attr_longtable(rq1_tbl, "Sex",        "rq1_odds_sex",    out_dir = "Tables/RQ1", caption_prefix = "RQ1: certainty \\(\\ge 3\\). ")
write_attr_longtable(rq1_tbl, "Income",     "rq1_odds_income", out_dir = "Tables/RQ1", caption_prefix = "RQ1: certainty \\(\\ge 3\\). ")
write_attr_longtable(rq1_tbl, agebin_label, "rq1_odds_agebin", out_dir = "Tables/RQ1", caption_prefix = "RQ1: certainty \\(\\ge 3\\). ")
write_attr_longtable(rq1_tbl, age5_label,   "rq1_odds_age5",   out_dir = "Tables/RQ1", caption_prefix = "RQ1: certainty \\(\\ge 3\\). ")

# summary -----------------------------------------------------------------

count_outperform <- function(gap_tbl) {
  gap_tbl %>%
    dplyr::filter(!arm %in% c("human_notimer","human_timepressure")) %>%
    summarise(
      models_total = dplyr::n(),
      better       = sum(mean_gap > 0, na.rm = TRUE),
      better_sig   = sum(ci_lo > 0,    na.rm = TRUE),
      worse        = sum(mean_gap < 0, na.rm = TRUE),
      worse_sig    = sum(ci_hi < 0,    na.rm = TRUE),
      .groups = "drop"
    )
}

count_from_glm <- function(h1_tab) {
  h1_tab %>%
    dplyr::filter(arm != "human_timepressure") %>%
    summarise(
      models_total = dplyr::n(),
      better_or    = sum(OR > 1, na.rm = TRUE),
      better_bh    = sum(OR > 1 & !is.na(p_BH) & p_BH < 0.05, na.rm = TRUE),
      .groups = "drop"
    )
}

sex_counts_rq1    <- count_outperform(gap_sex_rq1)
income_counts_rq1 <- count_outperform(gap_inc_rq1)
agebin_counts_rq1 <- count_outperform(gap_age_rq1)
age5_counts_rq1   <- count_outperform(gap_age5_rq1)

sex_glm_counts_rq1    <- count_from_glm(h1_sex_rq1$table)
income_glm_counts_rq1 <- count_from_glm(h1_inc_rq1$table)
agebin_glm_counts_rq1 <- count_from_glm(h1_age_rq1$table)
age5_glm_counts_rq1   <- count_from_glm(h1_age5_rq1$table)

sex_counts_rq1
income_counts_rq1
agebin_counts_rq1
age5_counts_rq1

sex_glm_counts_rq1
income_glm_counts_rq1
agebin_glm_counts_rq1
age5_glm_counts_rq1

