# Trial-level LPM with profile FE and two-way clustered SEs (profile × actor) for H1, H2, and RQ1.

# setup ----------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(stringr)
library(readr)
library(broom)
library(fixest)
library(kableExtra)
library(knitr)

set.seed(42)
fixest::setFixest_notes(FALSE)

dir.create("Tables/LPM_Robustness", recursive = TRUE, showWarnings = FALSE)

# inputs (created by earlier scripts) ----------------------------------------
human_scored    <- readRDS("data_derived/human_scored.rds")     # trial-level humans
llm_raw         <- readRDS("data_derived/llm_raw.rds")          # profile × model counts
combined_cert3  <- readRDS("data_derived/combined_cert3.rds")   # used only to confirm prof set for RQ1 if needed

# helpers --------------------------------------------------------------------
fmt_pp <- function(b, lo, hi, digits = 1) {
  # percentage points; NOTE: no % symbols in the cell text itself
  sprintf(paste0("%.", digits, "f (%.", digits, "f, %.", digits, "f)"),
          100*b, 100*lo, 100*hi)
}
fmt_p <- function(x, digits = 3, lt_threshold = 10^-digits) {
  ifelse(
    is.na(x), "",
    ifelse(x < lt_threshold,
           paste0("<", formatC(lt_threshold, format = "f", digits = digits)),
           formatC(x, format = "f", digits = digits))
  )
}
sanitize_latex <- function(x) {
  x %>%
    stringr::str_replace_all("×", "\\\\(\\\\times\\\\)") %>%  # 8×22B -> 8 \( \times \) 22B
    stringr::str_replace_all("—", "---") %>%                  # em dash -> ---
    stringr::str_replace_all("–", "--") %>%                   # en dash -> --
    stringr::str_replace_all("%", "\\\\%") %>%                # escape % (IMPORTANT)
    stringr::str_replace_all("_", "\\\\_") %>%                # escape _
    stringr::str_replace_all("&", "\\\\&")                    # escape &
}
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

# attribute maps --------------------------------------------------------------
attr_map_h <- list(           # human trial-level 0/1 outcomes (overall = primary)
  "Sex"                   = "sex_correct_overall",
  "Income"                = "income_correct_overall",
  "Age (\\(\\ge 40\\))"   = "agebin_correct_overall",
  "Age (\\(\\pm 5\\) y)"  = "agein5_correct_overall"   # appendix metric
)
attr_map_m <- list(           # LLM counts -> build rates
  "Sex"                   = "successes_sex_overall",
  "Income"                = "successes_income_overall",
  "Age (\\(\\ge 40\\))"   = "successes_agebin_overall",
  "Age (\\(\\pm 5\\) y)"  = "successes_agein5_overall"
)

# data builders ---------------------------------------------------------------
build_humans <- function(dat, subset = c("main","rq1")) {
  subset <- match.arg(subset)
  dd <- dat
  if (subset == "rq1") {
    dd <- dd |> filter(!is.na(certainty), certainty >= 3)
  }
  dd |>
    transmute(
      profile_fe = as.character(profile_id),
      actor_id   = as.character(ResponseId),
      arm        = case_when(
        .data[["treatment"]] == "control"      ~ "human_notimer",
        .data[["treatment"]] == "timepressure" ~ "human_timepressure",
        TRUE ~ NA_character_
      ),
      sex   = .data[[attr_map_h[["Sex"]]]],
      inc   = .data[[attr_map_h[["Income"]]]],
      age40 = .data[[attr_map_h[["Age (\\(\\ge 40\\))"]]]],
      age5  = .data[[attr_map_h[["Age (\\(\\pm 5\\) y)"]]]],
      w     = 1,
      src   = "human"
    ) |>
    filter(!is.na(arm))
}

build_llms_weighted <- function(dat) {
  dat |>
    transmute(
      profile_fe = as.character(profile),
      actor_id   = as.character(arm),      # cluster by model id (arm label)
      arm        = as.character(arm),
      trials     = pmax(trials_overall, 0L),
      sex_rate   = ifelse(trials > 0, successes_sex_overall    / trials, NA_real_),
      inc_rate   = ifelse(trials > 0, successes_income_overall / trials, NA_real_),
      age40_rate = ifelse(trials > 0, successes_agebin_overall / trials, NA_real_),
      age5_rate  = ifelse(trials > 0, successes_agein5_overall / trials, NA_real_),
      w          = trials,
      src        = "llm"
    ) |>
    filter(trials > 0)
}

# stackers --------------------------------------------------------------------
stack_for_attr <- function(hum, llm_w, attr_label, include_llm = TRUE) {
  if (attr_label == "Sex") {
    yh <- hum$sex;     ym <- llm_w$sex_rate
  } else if (attr_label == "Income") {
    yh <- hum$inc;     ym <- llm_w$inc_rate
  } else if (attr_label == "Age (\\(\\ge 40\\))") {
    yh <- hum$age40;   ym <- llm_w$age40_rate
  } else if (attr_label == "Age (\\(\\pm 5\\) y)") {
    yh <- hum$age5;    ym <- llm_w$age5_rate
  } else stop("Unknown attribute label: ", attr_label)
  
  dh <- hum |>
    mutate(y = yh) |>
    select(profile_fe, actor_id, arm, y, w, src) |>
    filter(!is.na(y))
  
  if (!include_llm) return(dh)
  
  dm <- llm_w |>
    mutate(y = ym) |>
    select(profile_fe, actor_id, arm, y, w, src) |>
    filter(!is.na(y))
  
  bind_rows(dh, dm)
}

# model runner ---------------------------------------------------------------
run_lpm_profilefe_twocluster <- function(df, ref_arm = "human_notimer") {
  df <- df |> mutate(
    arm = relevel(factor(arm), ref = ref_arm),
    profile_fe = factor(profile_fe),
    actor_id   = factor(actor_id)
  )
  feols(y ~ arm | profile_fe, data = df,
        weights = ~ w,
        cluster = ~ profile_fe + actor_id)
}

tidy_lpm <- function(fit, attr_label, block_label, add_counts_df) {
  tt  <- broom::tidy(fit, conf.int = TRUE) |>
    dplyr::filter(stringr::str_detect(term, "^arm")) |>
    mutate(
      Arm = sub("^arm", "", term),
      `Coef (pp, 95% CI)` = fmt_pp(estimate, conf.low, conf.high),
      `p-value` = fmt_p(p.value, 3),
      is_htp = Arm == "human_timepressure"
    ) |>
    arrange(desc(is_htp), desc(estimate)) |>
    transmute(
      Attribute = attr_label,
      Block     = block_label,
      Arm,
      `Coef (pp, 95% CI)`,
      `p-value`
    )
  
  counts <- add_counts_df |>
    summarise(
      N_obs       = sum(ifelse(src == "human", 1, 0)),
      N_weighted  = sum(ifelse(src == "llm", w, 0)),
      Profiles_N  = n_distinct(profile_fe),
      Actors_N    = n_distinct(actor_id)
    )
  attr(tt, "counts") <- counts
  tt
}

# blocks: H1, H2, RQ1 --------------------------------------------------------
hum_main <- build_humans(human_scored, subset = "main")
hum_rq1  <- build_humans(human_scored, subset = "rq1")
llm_w    <- build_llms_weighted(llm_raw)

attributes_vec <- c("Sex", "Income", "Age (\\(\\ge 40\\))", "Age (\\(\\pm 5\\) y)")

run_block <- function(block = c("H1","H2","RQ1")) {
  block <- match.arg(block)
  results <- vector("list", length(attributes_vec)); names(results) <- attributes_vec
  
  for (attr_label in attributes_vec) {
    if (block == "H1") {
      df_attr <- stack_for_attr(hum_main, llm_w, attr_label, include_llm = TRUE) |>
        filter(arm != "human_timepressure" | src == "human")
      fit <- run_lpm_profilefe_twocluster(df_attr, ref_arm = "human_notimer")
      tab <- tidy_lpm(fit, attr_label, "H1: vs Human (no timer)", df_attr)
      
    } else if (block == "H2") {
      df_attr <- stack_for_attr(hum_main, llm_w, attr_label, include_llm = FALSE) |>
        filter(arm %in% c("human_notimer","human_timepressure"))
      fit <- run_lpm_profilefe_twocluster(df_attr, ref_arm = "human_notimer")
      tab <- tidy_lpm(fit, attr_label, "H2: Human TP vs NT", df_attr)
      
    } else if (block == "RQ1") {
      df_attr <- stack_for_attr(hum_rq1, llm_w, attr_label, include_llm = TRUE) |>
        filter(arm != "human_timepressure" | src == "human")
      fit <- run_lpm_profilefe_twocluster(df_attr, ref_arm = "human_notimer")
      tab <- tidy_lpm(fit, attr_label, "RQ1: certainty ≥3", df_attr)
    }
    results[[attr_label]] <- tab
  }
  results
}

h1_res  <- run_block("H1")
h2_res  <- run_block("H2")
rq1_res <- run_block("RQ1")

# export tables ---------------------------------------------------------------
export_block <- function(res_list, outfile_stub, caption_prefix) {
  all_tab <- bind_rows(res_list)
  for (attr_label in unique(all_tab$Attribute)) {
    df_attr <- all_tab |> filter(Attribute == attr_label)
    
    cap <- paste0(
      caption_prefix, sanitize_latex(attr_label),
      ". Linear probability model with profile fixed effects; ",
      "two-way clustered SEs by profile and actor. ",
      "Coefficients are percentage-point differences vs Human (no timer)."
    )
    
    out <- df_attr |>
      mutate(
        Attribute = sanitize_latex(Attribute),
        Block     = sanitize_latex(Block),
        Arm       = sanitize_latex(Arm),
        `Coef (pp, 95% CI)` = sanitize_latex(`Coef (pp, 95% CI)`),
        `p-value` = `p-value`
      ) |>
      # rename the header so % is escaped in LaTeX
      rename(`Coef (pp, 95\\% CI)` = `Coef (pp, 95% CI)`) |>
      select(Arm, `Coef (pp, 95\\% CI)`, `p-value`)
    
    write_longtable(
      out,
      file = file.path("Tables/LPM_Robustness",
                       sprintf("%s_%s.tex",
                               outfile_stub,
                               str_replace_all(attr_label, "[^A-Za-z0-9]+", "_"))),
      caption = cap,
      align = c("l","l","r")
    )
  }
}

export_block(h1_res,  "robustness_lpm_h1",  "Robustness (H1): ")
export_block(h2_res,  "robustness_lpm_h2",  "Robustness (H2): ")
export_block(rq1_res, "robustness_lpm_rq1", "Robustness (RQ1, certainty ≥3): ")
