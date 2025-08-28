# File Arguments:
MODEL_NAME <- "config13" # The name of the Model whose output is being evaluated
MODEL_OUTPUT_PATH <- "../output/13.led-base.billsum_clean_train_se3-led-1024-512.binary_blank_targets.1024_512_5_epochs.checkpoint-108471.csv" # place to look for model output csv
ALT_NAME <- "se3(1024)" # The name of the Alternate Model to evaluate against (or just "Gold" if comparing to gold data)
ALT_PATH <- "../output/0.led-base.billsum_clean_train_se3-led-1024-512.drop_blank_targets.1024_512_5_epochs.checkpoint-41832.csv" # place to look for gold data/alternate model data csv
BASE_NAME <- "Gold"
BASELINE_OUTPUT_PATH <- "gold_reference_metrics.csv"
ANALYSIS_PATH <- "tristograms-thirteen-1024base/" # place to write the plots and stats to. Must end with a "/"
HISTOGRAM_BINS <- 60
# How to use this script:
# --- 1. Adjust the File Arguments for the desired comparison between models (or between model and gold summaries)
# ------ 1a. Make sure to give descriptive names for MODEL_NAME and ALT_NAME and BASE_NAME as these will show up throughout all the analysis files
# --- 2. Give a name to ANALYSIS_PATH where everything should go. Make sure to include the final "/"
# --- 3. Adjust the number of histogram bins that you want
# --- 4. Ensure your working directory is correct and hit 'source' in Rstudio
# ---------- Some Notes ---------- #
# This script will generate visualizations for three-way comparisons between
# MODEL_OUTPUT_PATH, ALT_OUTPUT_PATH, BASE_OUTPUT_PATH
# ----------------------------------------------------------------------- #
# Do not change these variables. These represent suffixes we expect to read in from csv headers
library(tidyr)
library(ggplot2)
library(stringr)
library(dplyr)
library(pastecs)
library(effsize)


ALT_SUFFIX <- ".GOLD"
GEN_SUFFIX <- ".GEN"

long <- "(.+)\\(.+\\)"
SHORT_MODEL_NAME <- ifelse(str_detect(MODEL_NAME,long), str_extract(MODEL_NAME,long,group=1),MODEL_NAME)
SHORT_ALT_NAME <- ifelse(str_detect(ALT_NAME,long), str_extract(ALT_NAME,long,group=1),ALT_NAME)
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ------------------------- (0) Setup Variables ------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #


# LFTK readability and other metrics of gold and generated summaries on test partition
# These two CSVs must have to same columns
metrics <- list() #store metrics for generated summaries and alternate summaries
metrics[['gen']] <- read.csv(MODEL_OUTPUT_PATH)
metrics[['alt']] <- read.csv(ALT_PATH)
metrics[['base']] <- read.csv(BASELINE_OUTPUT_PATH)

# Remove the redundant '.GOLD' lftk columns if included in file headers at MODEL_OUTPUT_PATH then
# set column names properly for processing
# metrics$alt <- metrics$alt %>% slice(1:16)
metrics$gen <- metrics$gen %>% select(-ends_with(ALT_SUFFIX))
colnames(metrics$gen) <- str_replace(colnames(metrics$gen),GEN_SUFFIX,"")
colnames(metrics$alt) <- str_replace(colnames(metrics$alt),ALT_SUFFIX,"")
colnames(metrics$alt) <- str_replace(colnames(metrics$alt),GEN_SUFFIX,"")
colnames(metrics$base) <- str_replace(colnames(metrics$base),GEN_SUFFIX,"")
colnames(metrics$base) <- str_replace(colnames(metrics$base),ALT_SUFFIX,"")

# Refers to family of metrics. Includes Readability, Relevance, Factuality, etc. See README
family <- list()
family$wordsent <- c(
  "t_word",           
  "t_stopword",
  "t_punct",
  "t_syll",
  "t_syll2",
  "t_syll3",
  "t_uword",          
  "t_sent",
  "t_char"
)
family$readformula <- c( # Readability Metrics
  "fkre",
  "fkgl",
  "fogi",
  "smog",
  "cole",
  "auto"
)
family$worddiff <- c(
  "t_kup",
  "t_bry",
  "t_subtlex_us_zipf"
)
family$entity <- c(
  "t_n_ent_law"
)
family$relevance <- c( # Relevance Metrics
  "rouge1_precision",
  "rouge2_precision",
  "rougeL_precision",
  "rouge1_recall",
  "rouge2_recall",
  "rougeL_recall",
  "rouge1_fmeasure",
  "rouge2_fmeasure",  
  "rougeL_fmeasure",
  "bertscore_p",
  "bertscore_r",
  "bertscore_f1"
)
family$factuality <- c( # Factuality Metrics
  "align_score",
  "summac"
)

# ------------------------------------------------------------ #
# -------------------- (1) Generate plots -------------------- #
# ------------------------------------------------------------ #

# Generate Tristograms
for (feature in colnames(metrics$gen)) {
  for(fam in names(family)){
    if(!(feature %in% family[[fam]]) || !(feature %in% colnames(metrics$alt)) || !(feature %in% colnames(metrics$base))){
      next
    }
    plt <- 
      ggplot() + 
      geom_histogram(aes(x=metrics$gen[[feature]], fill=MODEL_NAME), alpha=0.70,bins=HISTOGRAM_BINS) +
      geom_histogram(aes(x=metrics$alt[[feature]], fill=ALT_NAME),alpha=0.60,bins=HISTOGRAM_BINS) +
      geom_histogram(aes(x=metrics$base[[feature]], fill=BASE_NAME),alpha=0.50,bins=HISTOGRAM_BINS) +
      # scale_x_continuous(breaks = seq(0,30,5)) +
      labs(
        x = 'Value', 
        y = "Count", 
        title= str_c(feature," for ",MODEL_NAME,", ",ALT_NAME," and ",BASE_NAME," Summaries")
      ) #+
    plt
    # Save plots in right place
    path_to_save <- str_c(ANALYSIS_PATH,"tristograms/",fam,"/")
    ggsave(str_c(path_to_save,feature,"_distribution_",MODEL_NAME,"_",ALT_NAME,"_and_",BASE_NAME,"_summaries.png"), plt, create.dir = TRUE)
    
  } # ----- End of Attributes loop
} # ----- End of features loop



# # Generate Boxplots
# for (feature in colnames(metrics$gen)) {
#   for(fam in names(family)){
#     if(!(feature %in% family[[fam]]) || !(feature %in% colnames(metrics$alt)) || !(feature %in% colnames(metrics$base))){
#       next
#     }
#     
#     # d <- cbind(metrics$gen, MODEL = rep(SHORT_MODEL_NAME,nrow(metrics$gen))) %>% 
#     #   rbind(
#     #     cbind(metrics$alt, MODEL = rep(SHORT_ALT_NAME,nrow(metrics$alt)))
#     #   ) %>% 
#     #   rbind(
#     #     cbind(metrics$base, MODEL = rep(BASE_NAME,nrow(metrics$base)))
#     #   )
#     
#     plt <- 
#       ggplot() + 
#         geom_boxplot(aes(x=metrics$alt[[feature]]), y="model2") +
#         geom_boxplot(aes(x=metrics$gen[[feature]], y="model1")) +
#         geom_boxplot(aes(x=metrics$base[[feature]]),y="model3") +
#       # scale_x_continuous(breaks = seq(0,30,5)) +
#       labs(
#         x = 'Value', 
#         y = "Count", 
#         title= str_c(feature," for ",SHORT_MODEL_NAME,", ",SHORT_ALT_NAME," and ",BASE_NAME," Summaries")
#       ) #+
#     plt
#     # Save plots in right place
#     path_to_save <- str_c(ANALYSIS_PATH,"boxplots/",fam,"/")
#     ggsave(str_c(path_to_save,feature,"_distribution_",MODEL_NAME,"_",ALT_NAME,"_and_",BASE_NAME,"_summaries.png"), plt, create.dir = TRUE)
#     
#   } # ----- End of Attributes loop
# } # ----- End of features loop


