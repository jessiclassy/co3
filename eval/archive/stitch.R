library(tidyr)
library(ggplot2)
library(stringr)
library(dplyr)
library(pastecs)


baseline_lftk <- read.csv("deliverable_2/pegasusbillsum_baseline_lftk.csv")
baseline_rouge <- read.csv("deliverable_2/pegasusbillsum_rouge_scores.txt",sep = "\t")
baseline_rouge <- baseline_rouge %>% slice(1:nrow(baseline_lftk))
baseline <- data.frame(X = baseline_lftk$X)

# Add Precisions
baseline$rouge1_precision <- 
  as.numeric(
    str_extract(baseline_rouge$rouge1,
                "precision=([\\d\\.e\\+\\-]+)",
                group = 1)
    )
baseline$rouge2_precision <- 
  as.numeric(
    str_extract(baseline_rouge$rouge2,
                "precision=([\\d\\.e\\+\\-]+)",
                group = 1)
    )
baseline$rougeL_precision <- 
  as.numeric(
    str_extract(baseline_rouge$rougeL,
                "precision=([\\d\\.e\\+\\-]+)",
                group = 1)
    )

# Add Recalls
baseline$rouge1_recall <- 
  as.numeric(
    str_extract(baseline_rouge$rouge1,
                "recall=([\\d\\.e\\+\\-]+)",
                group = 1)
  )
baseline$rouge2_recall <- 
  as.numeric(
    str_extract(baseline_rouge$rouge2,
                "recall=([\\d\\.e\\+\\-]+)",
                group = 1)
  )
baseline$rougeL_recall <- 
  as.numeric(
    str_extract(baseline_rouge$rougeL,
                "recall=([\\d\\.e\\+\\-]+)",
                group = 1)
  )

# Add F measures
baseline$rouge1_fmeasure <- 
  as.numeric(
    str_extract(baseline_rouge$rouge1,
                "fmeasure=([\\d\\.e\\+\\-]+)",
                group = 1)
  )
baseline$rouge2_fmeasure <- 
  as.numeric(
    str_extract(baseline_rouge$rouge2,
                "fmeasure=([\\d\\.e\\+\\-]+)",
                group = 1)
  )
baseline$rougeL_fmeasure <- 
  as.numeric(
    str_extract(baseline_rouge$rougeL,
                "fmeasure=([\\d\\.e\\+\\-]+)",
                group = 1)
  )

# Attach the readability metrics in a pretty order
baseline <- baseline %>% 
  cbind(baseline_lftk %>% select(-c("X","summary_generated"))) %>% 
  cbind(baseline_lftk %>% select(c("summary_generated")))

write.csv(baseline,file = "pegasusbillsum_baseline_ALL_metrics.csv",row.names = F)

