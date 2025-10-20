
# setup -------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(stringr)
library(broom)
library(fixest)
library(ggplot2)
library(scales)
library(forcats)
library(knitr)
library(kableExtra)

dir.create("Figures", showWarnings = FALSE)
dir.create("Tables", showWarnings = FALSE)

set.seed(42)
fixest::setFixest_notes(FALSE)

combined_main <- readRDS("data_derived/combined_main.rds")   # H1/H2

# descriptive: raw accuracy ------------------------------------------------

raw_overall_accuracy <- combined_main %>%
  group_by(arm) %>%
  summarise(
    trials_total             = sum(trials_overall,           na.rm = TRUE),
    successes_sex_total      = sum(successes_sex_overall,    na.rm = TRUE),
    successes_agebin_total   = sum(successes_agebin_overall, na.rm = TRUE),
    successes_agein5_total   = sum(successes_agein5_overall, na.rm = TRUE),
    successes_income_total   = sum(successes_income_overall, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    acc_sex_overall     = successes_sex_total    / trials_total,
    acc_agebin_overall  = successes_agebin_total / trials_total,
    acc_agein5_overall  = successes_agein5_total / trials_total,
    acc_income_overall  = successes_income_total / trials_total
  ) %>%
  select(
    arm, trials_total,
    acc_sex_overall, acc_agebin_overall, acc_agein5_overall, acc_income_overall
  ) %>%
  arrange(arm)

make_raw_plot <- function(raw_tbl, acc_col, x_label, file_stub) {
  df <- raw_tbl %>%
    select(arm, !!sym(acc_col)) %>%
    rename(acc = !!sym(acc_col)) %>%
    mutate(
      arm_label = case_when(
        arm == "human_notimer"      ~ "Human (no timer)",
        arm == "human_timepressure" ~ "Human (time pressure)",
        TRUE                        ~ arm
      ),
      is_human    = arm %in% c("human_notimer","human_timepressure"),
      legend_class = ifelse(is_human, "Human", "LLM")
    ) %>%
    arrange(desc(acc)) %>%
    mutate(y_fac = factor(arm_label, levels = rev(arm_label)))
  
  p <- ggplot(df, aes(x = acc, y = y_fac, color = legend_class)) +
    geom_point(size = 2.5, shape = 16) +                      # circle for all
    scale_x_continuous(labels = scales::percent_format(accuracy = 0.1),
                       limits = c(0, 1)) +
    scale_color_manual(
      values = c("Human" = "#1b73e8", "LLM" = "#444444"),
      breaks = c("Human", "LLM"),
      name = NULL
    ) +
    labs(x = x_label, y = NULL) +
    theme_classic(base_size = 12) +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.position  = "bottom",
      legend.direction = "horizontal",
      plot.title       = element_blank(),
      plot.subtitle    = element_blank()
    )
  
  ggsave(sprintf("Figures/%s.png", file_stub), p, width = 8, height = 9, dpi = 300)
  p
}

p_raw_sex     <- make_raw_plot(raw_overall_accuracy, "acc_sex_overall",
                               "Raw overall accuracy (sex)",
                               "fig_raw_acc_sex")

p_raw_agebin  <- make_raw_plot(raw_overall_accuracy, "acc_agebin_overall",
                               "Raw overall accuracy (age ≥40)",
                               "fig_raw_acc_agebin")

p_raw_age5    <- make_raw_plot(raw_overall_accuracy, "acc_agein5_overall",
                               "Raw overall accuracy (age within 5 years)",
                               "fig_raw_acc_age5")

p_raw_income  <- make_raw_plot(raw_overall_accuracy, "acc_income_overall",
                               "Raw overall accuracy (income)",
                               "fig_raw_acc_income")


# helper functions --------------------------------------------------------

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

# h1/h2 ----------------------------------------------------------------------

df <- combined_main

# SEX
h1_sex  <- fit_h1_one(df,
                      successes_col = "successes_sex_overall",
                      trials_col    = "trials_overall")
gap_sex <- paired_profile_gap(df,
                              successes_col = "successes_sex_overall",
                              trials_col    = "trials_overall")

# AGE (bin <40 vs ≥40)
h1_age  <- fit_h1_one(df,
                      successes_col = "successes_agebin_overall",
                      trials_col    = "trials_overall")
gap_age <- paired_profile_gap(df,
                              successes_col = "successes_agebin_overall",
                              trials_col    = "trials_overall")

# AGE (in 5)
h1_age5 <- fit_h1_one(df,
  successes_col = "successes_agein5_overall",
  trials_col    = "trials_overall"
)
gap_age5 <- paired_profile_gap(df,
  successes_col = "successes_agein5_overall",
  trials_col    = "trials_overall"
)

# INCOME (3-level, scored 0/1 vs truth)
h1_inc  <- fit_h1_one(df,
                      successes_col = "successes_income_overall",
                      trials_col    = "trials_overall")
gap_inc <- paired_profile_gap(df,
                              successes_col = "successes_income_overall",
                              trials_col    = "trials_overall")

# Odds ratios (model vs human_notimer) with BH-adjusted p-values:
h1_sex$table   %>% arrange(p_BH)
h1_age$table   %>% arrange(p_BH)
h1_inc$table   %>% arrange(p_BH)

# Paired profile gaps (pp) with 95% bootstrap CIs:
gap_sex %>% arrange(desc(mean_gap))
gap_age %>% arrange(desc(mean_gap))
gap_age5 %>% arrange(desc(mean_gap))
gap_inc %>% arrange(desc(mean_gap))

# h1/h2 figures: paired profile gaps (pp) vs human_notimer ---------------------

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

p_age40 <- make_gap_plot(gap_age,  "fig_agebin_gap")
p_age5  <- make_gap_plot(gap_age5, "fig_age5_gap")
p_sex   <- make_gap_plot(gap_sex,  "fig_sex_gap")
p_inc   <- make_gap_plot(gap_inc,  "fig_income_gap")

# h1/H2 tables: log-odds --------------------------------------------------

fmt_or <- function(OR, lo, hi, digits = 2) {
  sprintf(paste0("%.", digits, "f (%.", digits, "f, %.", digits, "f)"),
          OR, lo, hi)
}

add_attr <- function(tab, attr) tab %>% mutate(attribute = attr, .before = 1)

agebin_label <- "Age (\\( \\ge 40 \\))"
age5_label   <- "Age (\\( \\pm 5 \\) years)"

t_sex   <- add_attr(h1_sex$table,  "Sex")
t_age   <- add_attr(h1_age$table,  agebin_label)
t_age5  <- add_attr(h1_age5$table, age5_label)   # NEW
t_inc   <- add_attr(h1_inc$table,  "Income")

options(scipen = 999)

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

h1_tbl <- bind_rows(t_sex, t_inc, t_age, t_age5) %>%
  mutate(
    arm_label = case_when(
      arm == "human_timepressure" ~ "Human (time pressure)",
      TRUE                        ~ arm
    ),
    `Odds ratio (95% CI)` = fmt_or(OR, OR_lo, OR_hi),  # already fixed-decimal via sprintf
    `p-value`            = fmt_p(p.value, 3),          # <-- no scientific notation
    `q-value (BH)`       = ifelse(is.na(p_BH), "", fmt_p(p_BH, 3))
  ) %>%
  select(
    Attribute = attribute,
    Arm = arm_label,
    `Odds ratio (95% CI)`,
    `p-value`,
    `q-value (BH)`
  ) %>%
  arrange(Attribute, factor(Arm, levels = c("Human (time pressure)")), Arm)

extract_or_num <- function(or_str) {
  as.numeric(stringr::str_match(or_str, "^([0-9]+\\.?[0-9]*)")[,2])
}

sanitize_latex <- function(x) {
  x %>%
    stringr::str_replace_all("×", "\\\\(\\\\times\\\\)") %>%  # 8×22B -> 8 \( \times \) 22B
    stringr::str_replace_all("—", "---") %>%                  # em dash -> LaTeX em-dash
    stringr::str_replace_all("–", "--")                       # en dash -> LaTeX en-dash
}

write_attr_longtable <- function(df, attr, file_stub) {
  df_attr <- df %>%
    dplyr::filter(Attribute == attr) %>%
    dplyr::mutate(
      is_htp = Arm == "Human (time pressure)",
      OR_num = extract_or_num(`Odds ratio (95% CI)`),
      Attribute = sanitize_latex(Attribute),
      Arm       = sanitize_latex(Arm),
      `Odds ratio (95% CI)` = sanitize_latex(`Odds ratio (95% CI)`),
      `p-value`   = `p-value`,
      `q-value (BH)` = `q-value (BH)`
    ) %>%
    dplyr::arrange(dplyr::desc(is_htp), dplyr::desc(OR_num)) %>%
    dplyr::select(-is_htp, -OR_num)
  
  cap <- paste0(
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
    save_kable(paste0("Tables/h1_h2_odds_", file_stub, ".tex"))
}

write_attr_longtable(h1_tbl, "Sex",     "sex")
write_attr_longtable(h1_tbl, "Income",  "income")
write_attr_longtable(h1_tbl, agebin_label, "agebin")
write_attr_longtable(h1_tbl, age5_label, "age5")

# h1 summary: paired profile gaps ------------------------------------------

count_outperform <- function(gap_tbl) {
  gap_tbl %>%
    dplyr::filter(!arm %in% c("human_notimer","human_timepressure")) %>%
    summarise(
      models_total   = dplyr::n(),
      better         = sum(mean_gap > 0, na.rm = TRUE),     # mean advantage > 0
      better_sig     = sum(ci_lo > 0,    na.rm = TRUE),     # 95% CI entirely > 0
      worse          = sum(mean_gap < 0, na.rm = TRUE),
      worse_sig      = sum(ci_hi < 0,    na.rm = TRUE),
      .groups = "drop"
    )
}

sex_counts    <- count_outperform(gap_sex)
income_counts <- count_outperform(gap_inc)
agebin_counts <- count_outperform(gap_age)
age5_counts   <- count_outperform(gap_age5)

sex_counts
income_counts
agebin_counts
age5_counts


# h1 summary: log odds -------------------------------------------------

count_from_glm <- function(h1_tab) {
  h1_tab %>%
    dplyr::filter(arm != "human_timepressure") %>%     # exclude human contrast
    summarise(
      models_total = dplyr::n(),
      better_or    = sum(OR > 1, na.rm = TRUE),        # OR > 1
      better_bh    = sum(OR > 1 & !is.na(p_BH) & p_BH < 0.05, na.rm = TRUE),
      .groups = "drop"
    )
}

sex_glm_counts    <- count_from_glm(h1_sex$table)
income_glm_counts <- count_from_glm(h1_inc$table)
agebin_glm_counts <- count_from_glm(h1_age$table)
age5_glm_counts   <- count_from_glm(h1_age5$table)

sex_glm_counts
income_glm_counts
agebin_glm_counts
age5_glm_counts

# h2 summary --------------------------------------------------------------

# Paired-by-profile percentage-point gaps (Human TP vs Human no timer)
h2_pp <- bind_rows(
  gap_sex  %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Sex",    pp = 100*mean_gap, lo = 100*ci_lo, hi = 100*ci_hi),
  gap_inc  %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Income", pp = 100*mean_gap, lo = 100*ci_lo, hi = 100*ci_hi),
  gap_age  %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Age ≥40", pp = 100*mean_gap, lo = 100*ci_lo, hi = 100*ci_hi),
  gap_age5 %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Age ±5y", pp = 100*mean_gap, lo = 100*ci_lo, hi = 100*ci_hi)
)

# FE-GLM odds ratios (Human TP vs Human no timer)
h2_or <- bind_rows(
  h1_sex$table  %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Sex",    OR, OR_lo, OR_hi, p = p.value),
  h1_inc$table  %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Income", OR, OR_lo, OR_hi, p = p.value),
  h1_age$table  %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Age ≥40", OR, OR_lo, OR_hi, p = p.value),
  h1_age5$table %>% filter(arm == "human_timepressure") %>% transmute(attribute = "Age ±5y", OR, OR_lo, OR_hi, p = p.value)
)

h2_pp
h2_or

