# File Arguments:
MODEL_NAME <- "wugNATSS-pegasus(on-unsimp)" # The name of the Model whose output is being evaluated
MODEL_OUTPUT_PATH <- "../output/deliverable_4/wugNATSS-pegasus/eval_on_unsimp.csv" # place to look for model output csv
ALT_NAME <- "Pegasus-Baseline" # The name of the Alternate Model to evaluate against (or just "Gold" if comparing to gold data)
ALT_PATH <- "../output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv" # place to look for gold data/alternate model data csv
ANALYSIS_PATH <- "analysis-temporary/" # place to write the plots and stats to. Must end with a "/"
HISTOGRAM_BINS <- 60 # number of bins to display in the histograms
# How to use this script:
# --- 1. Adjust the File Arguments for the desired comparison between models (or between model and gold summaries)
# ------ 1a. Make sure to give descriptive names for MODEL_NAME and ALT_NAME as these will show up throughout all the analysis files
# --- 2. Give a name to ANALYSIS_PATH where everything should go. Make sure to include the final "/"
# --- 3. Adjust the number of histogram bins that you want
# --- 4. Ensure your working directory is correct and hit 'source' in Rstudio
# ---------- Some Notes ---------- #
# - This is a file for generating model analysis plots for ling 573 project
# - All Analysis files will be placed neatly in their proper folder (stats, plots, qa) at the
# path specified by ANALYSIS_PATH 
# - About these arguments - "ALT_NAME" and "ALT_PATH" could be the output of a different
# model and not necessarily computed from gold/reference data. It's a bad naming convention, but
# when you want to compare two models, put the second model output in the "ALT_PATH" var and
# name it with "ALT_NAME" accordingly.
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

# This list will store 2 dataframes of metrics for 'gen' and 'alt' model output
metrics <- list()
metrics[["alt"]] <- read.csv(ALT_PATH)
metrics[['gen']] <- read.csv(MODEL_OUTPUT_PATH)

# Remove the redundant '.GOLD' lftk columns if included in file headers at MODEL_OUTPUT_PATH then
# set column names properly for processing
# metrics$alt <- metrics$alt %>% slice(1:16) # bit of code commented out in case we want a toy dataset
metrics$gen <- metrics$gen %>% select(-ends_with(ALT_SUFFIX))
colnames(metrics$alt) <- str_replace(colnames(metrics$alt),ALT_SUFFIX,"")
colnames(metrics$alt) <- str_replace(colnames(metrics$alt),GEN_SUFFIX,"")
colnames(metrics$gen) <- str_replace(colnames(metrics$gen),GEN_SUFFIX,"")
colnames(metrics$gen) <- str_replace(colnames(metrics$gen),ALT_SUFFIX,"")

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
  "bert_score"
)
family$factuality <- c( # Factuality Metrics
  "align_score",
  "summac"
)

# ------------------------------------------------------------ #
# -------------------- (1) Generate plots -------------------- #
# ------------------------------------------------------------ #

for (feature in colnames(metrics$gen)) {
  for(fam in names(family)){
    if(!(feature %in% family[[fam]]) || !(feature %in% colnames(metrics$alt))){
      next
    }
    plt <- 
    ggplot() + 
      geom_histogram(aes(x=metrics$gen[[feature]], fill=SHORT_MODEL_NAME), alpha=0.95,bins=HISTOGRAM_BINS) +
      geom_histogram(aes(x=metrics$alt[[feature]],fill=SHORT_ALT_NAME),alpha=0.55,bins=HISTOGRAM_BINS) +
      # scale_x_continuous(breaks = seq(0,30,5)) +
      labs(
        x = 'Value', 
        y = "Count", 
        title= str_c(feature," for ",SHORT_MODEL_NAME," and ",SHORT_ALT_NAME," Summaries")
      ) #+
    
    # Save plots in right place
    path_to_save <- str_c(ANALYSIS_PATH,"plots/",fam,"/")
    ggsave(str_c(path_to_save,feature,"_distribution_",MODEL_NAME,"_and_",ALT_NAME,"_summaries.png"),plt,create.dir = T)
    
  } # ----- End of Attributes loop
} # ----- End of lftk features loop


# -------------------------------------------------------------------------- #
# ------ (2) Run Statistical Tests for all metrics and write to file ------- #
# -------------------------------------------------------------------------- #

# Prep directories for populating analysis
unlink(str_c(ANALYSIS_PATH,"stats/t_tests"),recursive = T)
unlink(str_c(ANALYSIS_PATH,"stats/effect_size"),recursive = T)
dir.create(str_c(ANALYSIS_PATH,"stats/"),showWarnings = F)
dir.create(str_c(ANALYSIS_PATH,"stats/t_tests"),showWarnings = F)
dir.create(str_c(ANALYSIS_PATH,"stats/effect_size"),showWarnings = F)

# Run statistical tests for all non-excluded metric families
families_to_exclude <- c("entity","worddiff")
for(fam in names(family)){
  if((fam %in% families_to_exclude)){
    next
  }
  for(feature in family[[fam]]){
    if(!(feature %in% colnames(metrics$gen)) || !(feature %in% colnames(metrics$alt))){ # skip if either dataset is missing the feature
      next
    }
    model_1 <- metrics$gen[[feature]]
    model_2 <- metrics$alt[[feature]]
    
    t_test_result <- capture.output(t.test(model_1,model_2,paired = T))
    effect_size_result <- capture.output(cohen.d(model_1,model_2))
    
    # write t tests to file
    write(
      c(str_c("--- Feature = ",feature,": ",MODEL_NAME," vs ",ALT_NAME," Summaries"),
        str_c("model_1 = ",MODEL_NAME," | model_2 = ",ALT_NAME),t_test_result),
      file = str_c(ANALYSIS_PATH,"stats/t_tests/",fam,"_",MODEL_NAME,"_vs_",ALT_NAME,".txt"),
      append = T
    )
    
    # write cohen's d tests to file
    write(
      c(str_c("--- Feature = ",feature,": ",MODEL_NAME," vs ",ALT_NAME," Summaries"), effect_size_result),
      file = str_c(ANALYSIS_PATH,"stats/effect_size/",fam,"_",MODEL_NAME,"_vs_",ALT_NAME,".txt"),
      append = T
    )
  }
}

# Write descriptive statistics of 'gen' model output to file
result <- capture.output(metrics$gen %>% stat.desc())

write(
  c(str_c("--- Model: ",MODEL_NAME," ---"), result),
  file = str_c(ANALYSIS_PATH,"stats/",MODEL_NAME,"_descriptive_stats.txt"),
  append = T
)

# Write descriptive statistics of gold data/'alt' model output to file
result2 <- capture.output(metrics$alt %>% stat.desc())

write(
  c(str_c("--- Model: ",ALT_NAME," ---"), result2),
  file = str_c(ANALYSIS_PATH,"stats/",ALT_NAME,"_descriptive_stats.txt"),
  append = T
)


# ----------------------------------------------------------------------- #
# ---------- (3) Generate Samples for Qualitative Analysis of fkre ------ #
# ----------------------------------------------------------------------- #

# Add quantile information to dataframes for filtering
set.seed(20)
num_quantiles = 5
gold_quantiles <- metrics$alt %>% mutate(quantile = ntile(fkre,num_quantiles))
gen_quantiles <- metrics$gen %>% mutate(quantile = ntile(fkre,num_quantiles))

# Make a list of fkre quantiles for gold summaries fkre and model output fkre
smps_gold <- c()
smps_gen <- c()
for(i in 1:num_quantiles){
  df_gold <- gold_quantiles %>% filter(quantile == i)
  df_gen <- gen_quantiles %>% filter(quantile == i)
  smps_gold <- c(smps_gold,sample(df_gold$X,1))
  smps_gen <- c(smps_gen,sample(df_gen$X,1))
}

t_gold <- gold_quantiles %>% filter(X %in% smps_gold)
t2_gold <- t_gold %>% left_join(gen_quantiles,by="X",suffix = c(str_c(".", ALT_NAME),str_c(".", MODEL_NAME)))

t_gen <- gen_quantiles %>% filter(X %in% smps_gen)
t2_gen <- t_gen %>% left_join(gold_quantiles,by="X",suffix = c(str_c(".",MODEL_NAME),str_c(".",ALT_NAME)))


# Prep directories for writing analysis to files
unlink(str_c(ANALYSIS_PATH,"qa"),recursive = T)
dir.create(str_c(ANALYSIS_PATH,"qa"),showWarnings = F)

write.csv(t2_gold,file = str_c(ANALYSIS_PATH,"qa/fkre_quantiles_",ALT_NAME,".csv"),row.names = F)
write.csv(t2_gen,file = str_c(ANALYSIS_PATH,"qa/fkre_quantiles_",MODEL_NAME,".csv"),row.names = F)


