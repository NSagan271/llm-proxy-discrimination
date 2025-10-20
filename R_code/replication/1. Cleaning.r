
# packages ----------------------------------------------------------------

library(readxl)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

# data loading and exclusions ---------------------------------------------

data <- read_excel("Attribute Inference_September 29, 2025_21.34.xlsx")
data <- data[-1, ]

data <- data %>%
  filter(
    Q1.2 == "I agree to participate",
    Q2.1 != "Under 18",
    Q2.2 != "I do not reside in the United States",
    Q3.1 == "Somewhat disagree"
  )

# reshape to one row per respondent x profile ----------------------------

human_long <- data %>%
  # keep only the needed respondent IDs/arm and the profile columns
  select(
    ResponseId,
    treatment,
    matches("^\\d+_(Q7\\.3_1|Q7\\.4|Q7\\.5|Q7\\.7|Q7\\.6_Page Submit)$")
  ) %>%
  # pivot: split "1_Q7.3_1" into loop_index=1, raw_var="Q7.3_1"
  pivot_longer(
    cols = matches("^\\d+_"),
    names_to = c("loop_index", "raw_var"),
    names_pattern = "^(\\d+)_(.+)$",
    values_to = "value"
  ) %>%
  mutate(loop_index = as.integer(loop_index)) %>%
  # rename the raw_var values to tidy column names
  mutate(raw_var = recode(raw_var,
                          "Q7.3_1" = "age_guess",
                          "Q7.4" = "sex_guess",
                          "Q7.5" = "income_guess",
                          "Q7.7" = "certainty",
                          "Q7.6_Page Submit" = "inferencepage_timer")) %>%
  # spread back to one row per respondent × profile
  pivot_wider(names_from = raw_var, values_from = value)

# join profile_id --------------------------------------------------------

lookup <- read_csv("lookup.csv") %>%
  mutate(loop_index = as.integer(loop_index))  # cols: loop_index, profile

human_long <- human_long %>%
  left_join(lookup, by = "loop_index") %>%
  rename(profile_id = profile) %>% 
  select(-loop_index)

# drop loops not shown to respondent ---------------------------------

human_long <- human_long %>%
  mutate(viewed = if_else(!is.na(inferencepage_timer), 1L, 0L))

human_long <- human_long %>% filter(viewed == 1)

# recode variables --------------------------------------------------------

human_long <- human_long %>%
  mutate(
    inferencepage_timer = suppressWarnings(as.numeric(inferencepage_timer)),
    age_guess = suppressWarnings(as.numeric(age_guess)),
    income_guess = case_when(
      income_guess == "Less than $30,000"    ~ "low",
      income_guess == "$30,000 to $60,000"  ~ "middle",
      income_guess == "Above $60,000"       ~ "high",
      TRUE ~ NA_character_
    ),
    sex_guess = case_when(
      sex_guess == "Male"   ~ "male",
      sex_guess == "Female" ~ "female",
      TRUE ~ NA_character_
    )
  )

# certainty: map text -> 1..5 (NA stays NA), keep raw for audit
lvls <- c("Not at all certain",
          "Slightly certain",
          "Moderately certain",
          "Very certain",
          "Absolutely certain")

human_long <- human_long %>%
  mutate(certainty = as.integer(match(certainty, lvls)))

# comparison with ground truth --------------------------------------------

gt <- read_csv("ground_truth.csv") %>%
  rename(profile_id = profile) %>%
  mutate(
    true_age = as.numeric(true_age),
    true_age40 = if_else(!is.na(true_age), as.integer(true_age >= 40), NA_integer_)
  )

# join and score
human_scored <- human_long %>%
  left_join(gt %>% select(profile_id, true_sex, true_income, true_age, true_age40),
            by = "profile_id") %>%
  mutate(
    # answered indicators
    sex_answered    = !is.na(sex_guess),
    age_answered    = !is.na(age_guess),
    income_answered = !is.na(income_guess),
    
    # age bin (<40 vs >=40)
    age40_guess = if_else(age_answered, as.integer(age_guess >= 40), NA_integer_),
    
    # correctness: answered-only (NA if no answer)
    sex_correct_answered    = if_else(sex_answered,    as.integer(sex_guess    == true_sex),    NA_integer_),
    agebin_correct_answered  = if_else(age_answered,    as.integer(age40_guess  == true_age40),    NA_integer_),
    income_correct_answered = if_else(income_answered, as.integer(income_guess == true_income), NA_integer_),
    
    # correctness: overall (missing answers counted as 0)
    sex_correct_overall    = if_else(sex_answered,    as.integer(sex_guess    == true_sex),    0L),
    agebin_correct_overall  = if_else(age_answered,    as.integer(age40_guess  == true_age40),    0L),
    income_correct_overall = if_else(income_answered, as.integer(income_guess == true_income), 0L),
    
    # robustness: ±5 years
    agein5_correct_answered = if_else(age_answered & !is.na(true_age),
                                       as.integer(abs(age_guess - true_age) <= 5), NA_integer_),
    agein5_correct_overall  = if_else(age_answered & !is.na(true_age),
                                       as.integer(abs(age_guess - true_age) <= 5), 0L)
  )

# other trial level datasets --------------------------------------------

# trials with non-missing certainty (for H3a/H3b models)
human_trials_cert_nonmissing <- human_scored %>%
  dplyr::filter(!is.na(certainty))

# RQ1: keep only trials with certainty >= 3
human_trials_cert3 <- human_scored %>%
  dplyr::filter(!is.na(certainty) & certainty >= 3)

# aggregate to profile x arm ----------------------------------------------

aggregate_humans <- function(trials) {
  trials %>%
    group_by(profile_id, treatment) %>%
    summarise(
      # denominators
      trials_overall          = n(),
      trials_answered_sex     = sum(sex_answered,    na.rm = TRUE),
      trials_answered_age  = sum(age_answered,    na.rm = TRUE),
      trials_answered_income  = sum(income_answered, na.rm = TRUE),
      
      # numerators: overall and answered-only
      successes_sex_overall       = sum(sex_correct_overall,    na.rm = TRUE),
      successes_sex_answered      = sum(sex_correct_answered,   na.rm = TRUE),
      
      successes_agebin_overall    = sum(agebin_correct_overall, na.rm = TRUE),
      successes_agebin_answered   = sum(agebin_correct_answered,na.rm = TRUE),
      
      successes_income_overall    = sum(income_correct_overall, na.rm = TRUE),
      successes_income_answered   = sum(income_correct_answered,na.rm = TRUE),
      
      # appendix: ±5y robustness for age
      successes_agein5_overall    = sum(agein5_correct_overall, na.rm = TRUE),
      successes_agein5_answered   = sum(agein5_correct_answered,na.rm = TRUE),
      
      .groups = "drop"
    ) %>%
    mutate(
      arm = case_when(
        treatment == "timepressure" ~ "human_timepressure",
        treatment == "control"      ~ "human_notimer"
      )
    ) %>%
    select(
      profile = profile_id, arm,
      trials_overall,
      successes_sex_overall,     successes_agebin_overall,     successes_agein5_overall, successes_income_overall,
      trials_answered_sex,       successes_sex_answered,
      trials_answered_age,    successes_agebin_answered,    successes_agein5_answered,
      trials_answered_income,    successes_income_answered
    ) %>%
    arrange(profile, arm)
}

human_agg_main  <- aggregate_humans(human_scored)
human_agg_cert3 <- aggregate_humans(human_trials_cert3)

# join llm data ----------------------------------------------

llm_raw <- read_csv("llm_human_experiment_results.csv") # Note: "very high" was already merged into "high"

rename_arms <- function(x) {
  x <- str_remove(x, "_temp_0\\.0_top_p_1\\.0$")  # drop suffix
  case_when(
    str_starts(x, "openai_gpt-5-nano")            ~ "GPT-5 Nano",
    str_starts(x, "openai_gpt-5")                 ~ "GPT-5",
    str_starts(x, "openai_gpt-4.1")               ~ "GPT-4.1",
    str_starts(x, "openai_o4-mini-high")          ~ "o4-mini-high",
    str_starts(x, "openai_o3-pro")                ~ "o3-pro",
    str_starts(x, "openai_o3")                    ~ "o3",
    
    str_starts(x, "anthropic_claude-3.5-haiku")   ~ "Claude 3.5 Haiku",
    str_starts(x, "anthropic_claude-sonnet-4")    ~ "Claude Sonnet 4",
    str_starts(x, "anthropic_claude-opus-4.1")    ~ "Claude Opus 4.1",
    
    str_starts(x, "google_gemini-2.5-pro")        ~ "Gemini 2.5 Pro",
    str_starts(x, "google_gemini-2.5-flash")      ~ "Gemini 2.5 Flash",
    str_starts(x, "google_gemma-3-27b-it")        ~ "Gemma 3 27B",
    str_starts(x, "google_gemma-3-12b-it")        ~ "Gemma 3 12B",
    str_starts(x, "google_gemma-2-27b-it")        ~ "Gemma 2 27B",
    
    str_starts(x, "meta-llama_llama-3.3-70b-instruct") ~ "Llama 3.3 70B",
    str_starts(x, "meta-llama_llama-3.1-405b-instruct")~ "Llama 3.1 405B",
    str_starts(x, "meta-llama_llama-3.1-8b-instruct")  ~ "Llama 3.1 8B",
    
    str_starts(x, "mistralai_mistral-large-2411")  ~ "Mistral Large 24.11",
    str_starts(x, "mistralai_mixtral-8x22b-instruct") ~ "Mixtral 8×22B",
    
    str_starts(x, "deepseek_deepseek-r1-0528")     ~ "DeepSeek R1",
    str_starts(x, "deepseek_deepseek-v3.1-terminus")~ "DeepSeek V3.1 Terminus",
    
    str_starts(x, "qwen_qwen3-max")                ~ "Qwen3 Max",
    str_starts(x, "qwen_qwen3-30b-a3b")            ~ "Qwen3 30B A3B",
    str_starts(x, "qwen_qwen-2.5-72b-instruct")    ~ "Qwen 2.5 72B",
    
    str_starts(x, "nvidia_llama-3.1-nemotron-70b-instruct") ~ "Nemotron 70B",
    
    str_starts(x, "cohere_command-r-plus-08-2024") ~ "Command-R+ (08-2024)",
    str_starts(x, "cohere_command-r-08-2024")      ~ "Command-R (08-2024)",
    str_starts(x, "cohere_command-r7b-12-2024")    ~ "Command-R 7B (12-2024)",
    
    str_starts(x, "amazon_nova-pro-v1")            ~ "Nova Pro",
    str_starts(x, "amazon_nova-lite-v1")           ~ "Nova Lite",
    str_starts(x, "amazon_nova-micro-v1")          ~ "Nova Micro",
    
    str_starts(x, "x-ai_grok-4")                   ~ "Grok-4",
    str_starts(x, "x-ai_grok-3")                   ~ "Grok-3",
    
    str_starts(x, "microsoft_phi-4")               ~ "Phi-4",
    
    TRUE ~ x
  )
}

llm_raw <- llm_raw %>%
  mutate(arm = rename_arms(arm))

agg_cols <- names(human_agg_main)

llm_aligned <- llm_raw %>%
  mutate(
    profile = as.character(profile),
    arm     = as.character(arm)
  ) %>%
  select(all_of(agg_cols))

combined_main  <- bind_rows(human_agg_main,  llm_aligned)

combined_cert3 <- bind_rows(human_agg_cert3, llm_aligned)

# Identify profiles that still have at least one human_notimer row
keep_profiles <- combined_cert3 %>%
  group_by(profile) %>%
  summarise(has_human_notimer = any(arm == "human_notimer"), .groups = "drop") %>%
  filter(has_human_notimer) %>%
  pull(profile)

# Filter the combined_cert3 table to those profiles only (across all arms), per prereg
combined_cert3 <- combined_cert3 %>%
  filter(profile %in% keep_profiles)

# save datasets for analysis --------------------------------------------

dir.create("data_derived", showWarnings = FALSE)

saveRDS(combined_main,   "data_derived/combined_main.rds")   # H1/H2
saveRDS(human_scored,    "data_derived/human_scored.rds")    # H3a/H3b
saveRDS(llm_raw,         "data_derived/llm_raw.rds")         # H3c
saveRDS(combined_cert3,  "data_derived/combined_cert3.rds")  # RQ1 (cert ≥ 3)
