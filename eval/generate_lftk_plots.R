# File for generating model analysis plots for ling 573 project
# File Arguments:
MODEL_NAME <- "PegasusBillSum" # The name of the Model whose output is being evaluated
GOLD_NAME <- "PegasusBillSumBaseline" # The name of the Alternate Model to evaluate against (or just "Gold" if comparing to gold data)
MODEL_OUTPUT_PATH <- "../output/deliverable_3/pegasusbillsum_clean_se3_t5_simple_toy.csv" # place to look for model output csv
GOLD_PATH <- "../output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv" # place to look for gold data/alternate model data csv
ANALYSIS_PATH <- "deliverable_3/baseline_vs_pegasusbillsum/" # place to write the plots and tests to. Must end with a "/"
HISTOGRAM_BINS <- 60
# Note about these arguments - "GOLD_NAME" and "GOLD_PATH" could be the output of a different
# model and not necessarily computed from reference data. It's a bad naming convention, but
# when you want to compare two models, put the second model output in the "GOLD_PATH" var and
# name it with "GOLD_NAME" accordingly.
# ----------------------------------------------------------------------- #
# Do not change these variables. These represent suffixes we expect to read in from csv headers
GOLD_SUFFIX <- ".GOLD"
GEN_SUFFIX <- ".GEN"
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ------------------------- (0) Setup Variables ------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
library(tidyr)
library(ggplot2)
library(stringr)
library(dplyr)
library(pastecs)

# LFTK readability and other metrics of gold and generated summaries on test partition
# These two CSVs must have to same columns
lftk <- list()
lftk[["gold"]] <- read.csv(GOLD_PATH)
lftk[['gen']] <- read.csv(MODEL_OUTPUT_PATH)

# Remove the redundant '.GOLD' lftk columns if included in file headers at MODEL_OUTPUT_PATH then
# set column names properly for processing
lftk$gold <- lftk$gold %>% slice(1:16)
lftk$gen <- lftk$gen %>% select(-ends_with(GOLD_SUFFIX))
colnames(lftk$gold) <- str_replace(colnames(lftk$gold),GOLD_SUFFIX,"")
colnames(lftk$gen) <- str_replace(colnames(lftk$gen),GEN_SUFFIX,"")

# Refers to family of metrics. Includes LFTK and ROUGE. See README
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
family$readformula <- c(
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
family$rouge <- c(
  "rouge1_precision",
  "rouge2_precision",
  "rougeL_precision",
  "rouge1_recall",
  "rouge2_recall",
  "rougeL_recall",
  "rouge1_fmeasure",
  "rouge2_fmeasure",  
  "rougeL_fmeasure"
)

# ------------------------------------------------------------ #
# -------------------- (1) Generate plots -------------------- #
# ------------------------------------------------------------ #

for (feature in colnames(lftk$gen)) {
  for(fam in names(family)){
    if(!(feature %in% family[[fam]]) || !(feature %in% colnames(lftk$gold))){
      next
    }
    plt <- 
    ggplot() + 
      geom_histogram(aes(x=lftk$gen[[feature]], fill=MODEL_NAME), alpha=0.95,bins=HISTOGRAM_BINS) +
      geom_histogram(aes(x=lftk$gold[[feature]],fill=GOLD_NAME),alpha=0.55,bins=HISTOGRAM_BINS) +
      # scale_x_continuous(breaks = seq(0,30,5)) +
      labs(x = 'Value', y = "Count", title= str_c(feature," for ",MODEL_NAME," and ",GOLD_NAME," Summaries")) #+
      # scale_color_manual(name='Legend',
      #                    breaks=c(GOLD_NAME,MODEL_NAME),
      #                    values = c(GOLD_NAME="gold",MODEL_NAME="black"))
    
    # Save plots in right place
    path_to_save <- str_c(ANALYSIS_PATH,"plots/",fam,"/")
    ggsave(str_c(path_to_save,feature,"_distribution_",MODEL_NAME,"_and_",GOLD_NAME,"_summaries.png"),plt,create.dir = T)
    
  } # ----- End of Attributes loop
} # ----- End of lftk features loop


# -------------------------------------------------------------------------- #
# ---------------- (2) Run t.tests for ROUGE & LFTK metrics ---------------- #
# -------------------------------------------------------------------------- #

# Prep directories for populating analysis
batching_path = str_c(ANALYSIS_PATH,"stats/t_tests/readability_t_tests_grouped_by=t_char/")
unlink(str_c(ANALYSIS_PATH,"stats/t_tests"),recursive = T)
dir.create(str_c(ANALYSIS_PATH,"stats/"),showWarnings = F)
dir.create(str_c(ANALYSIS_PATH,"stats/t_tests"),showWarnings = F)
dir.create(batching_path,showWarnings = F)

# Get quantile-batched readformula t tests
num_quantiles = 5
gen_quantiles <- lftk$gen %>% mutate(quantile = ntile(t_char,num_quantiles))
gold_quantiles <- lftk$gold %>% mutate(quantile = ntile(t_char,num_quantiles))

for(feature in family$readformula){
  # Do t tests for each quantile
  for(q in 1:num_quantiles){
    model_1 <- (gen_quantiles %>% filter(quantile == q))[[feature]]
    model_2 <- (gold_quantiles %>% filter(quantile == q))[[feature]]

    result <- capture.output(t.test(model_1,model_2)) # cannot be paired because pairing is lost in quantile batching

    # write to file
    write(
      c(str_c("--- LFTK feature = ",feature,": ", MODEL_NAME," vs ",GOLD_NAME," Summaries - Quantile ", q, " of ", num_quantiles, " - Data batched by character count"),
        str_c("model_1 = ",MODEL_NAME," | model_2 = ",GOLD_NAME), result),
      file = str_c(batching_path,feature,"_by_groups.txt"),
      append = T
    )

  }
}

# 
families_to_exclude <- c("entity","worddiff")
for(fam in names(family)){
  if((fam %in% families_to_exclude)){
    next
  }
  for(feature in family[[fam]]){
    if(!(feature %in% colnames(lftk$gen)) || !(feature %in% colnames(lftk$gold))){ # skip if either dataset is missing the feature
      next
    }
    model_1 <- lftk$gen[[feature]]
    model_2 <- lftk$gold[[feature]]
    
    result <- capture.output(t.test(model_1,model_2,paired = T))
    
    # write to file
    write(
      c(str_c("--- Feature = ",feature,": ",MODEL_NAME," vs ",GOLD_NAME," Summaries"),
        str_c("model_1 = ",MODEL_NAME," | model_2 = ",GOLD_NAME),result),
      file = str_c(ANALYSIS_PATH,"stats/t_tests/",fam,"_",MODEL_NAME,"_vs_",GOLD_NAME,".txt"),
      append = T
    )
  }
}

# # Get wordsent t tests
# for(feature in family$wordsent){
#   
#   model_1 <- lftk$gen[[feature]]
#   model_2 <- lftk$gold[[feature]]
#   
#   result <- capture.output(t.test(model_1,model_2,paired = T))
#   
#   # write to file
#   write(
#     c(str_c("--- LFTK feature = ",feature,": ",MODEL_NAME," vs ",GOLD_NAME," Summaries"),
#       str_c("model_1 = ",MODEL_NAME," | model_2 = ",GOLD_NAME),result),
#     file = str_c(ANALYSIS_PATH,"stats/t_tests/wordsent_",MODEL_NAME,"_vs_",GOLD_NAME,".txt"),
#     append = T
#   )
# }
# 
# # Get full data readformula t tests
# for(feature in family$readformula){
#   
#   model_1 <- lftk$gen[[feature]]
#   model_2 <- lftk$gold[[feature]]
#   
#   result <- capture.output(t.test(model_1,model_2,paired = T))
# 
#   # write to file
#   write(
#     c(str_c("--- LFTK feature = ",feature,": ",MODEL_NAME," vs ",GOLD_NAME," Summaries"), 
#       str_c("model_1 = ",MODEL_NAME," | model_2 = ",GOLD_NAME), result),
#     file = str_c(ANALYSIS_PATH,"stats/t_tests/readformula_",MODEL_NAME,"_vs_",GOLD_NAME,".txt"),
#     append = T
#   )
# }
# 
# # Get ROUGE t tests
# for(feature in family$rouge){
# 
#   model_1 <- lftk$gen[[feature]]
#   model_2 <- lftk$gold[[feature]]
# 
#   result <- capture.output(t.test(model_1,model_2,paired = T))
# 
#   # write to file
#   write(
#     c(str_c("--- Feature = ",feature,": ",MODEL_NAME," vs ",GOLD_NAME," Summaries"), 
#       str_c("model_1 = ",MODEL_NAME," | model_2 = ",GOLD_NAME), result),
#     file = str_c(ANALYSIS_PATH,"stats/t_tests/rouge_",MODEL_NAME,"_vs_",GOLD_NAME,".txt"),
#     append = T
#   )
# }


# Get model summary
result <- capture.output(lftk$gen %>% stat.desc())

write(
  c(str_c("--- Model: ",MODEL_NAME," ---"), result),
  file = str_c(ANALYSIS_PATH,"stats/",MODEL_NAME,"_descriptive_stats.txt"),
  append = T
)

# Get gold/alternate model summary
result2 <- capture.output(lftk$gold %>% stat.desc())

write(
  c(str_c("--- Model: ",GOLD_NAME," ---"), result2),
  file = str_c(ANALYSIS_PATH,"stats/",GOLD_NAME,"_descriptive_stats.txt"),
  append = T
)


# ----------------------------------------------------------------------- #
# ---------- (3) Generate Samples for Qualitative Analysis of fkre ------ #
# ----------------------------------------------------------------------- #

# Add quantile information to dataframes for filtering
set.seed(20)
num_quantiles = 5
gold_quantiles <- lftk$gold %>% mutate(quantile = ntile(fkre,num_quantiles))
gen_quantiles <- lftk$gen %>% mutate(quantile = ntile(fkre,num_quantiles))

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
t2_gold <- t_gold %>% left_join(gen_quantiles,by="X",suffix = c(str_c(".", GOLD_NAME),str_c(".", MODEL_NAME)))

t_gen <- gen_quantiles %>% filter(X %in% smps_gen)
t2_gen <- t_gen %>% left_join(gold_quantiles,by="X",suffix = c(str_c(".",MODEL_NAME),str_c(".",GOLD_NAME)))


# Prep directories for writing analysis to files
unlink(str_c(ANALYSIS_PATH,"qa"),recursive = T)
dir.create(str_c(ANALYSIS_PATH,"qa"),showWarnings = F)

write.csv(t2_gold,file = str_c(ANALYSIS_PATH,"qa/fkre_quantiles_",GOLD_NAME,".csv"),row.names = F)
write.csv(t2_gen,file = str_c(ANALYSIS_PATH,"qa/fkre_quantiles_",MODEL_NAME,".csv"),row.names = F)






# Old plots I may want to use later

# Generate bar charts for readability metrics
# j <- lftk$gold %>% left_join(lftk$gen,by="X",suffix=c(".GOLD",".GEN"))
# bar <- j %>% pivot_longer(cols = colnames(j)[str_detect(colnames(j),"[(GOLD)(GEN)]")])
# bar <- bar %>% mutate(metric = str_extract(name,"([^.]*)\\.",group=1),
#                       summary_type = str_extract(name,"\\.(.*)",group=1))
# 
# bar %>% filter(metric %in% family$readformula) %>%
#   ggplot(aes(x = summary_type,y=value)) +
#   geom_boxplot() +
#   facet_wrap(~metric) +
#   coord_cartesian(ylim = c(-50, 100)) +
#   labs(x = "Summary Type",
#        y = "Value",
#        title = "Generated vs Gold Summaries")



