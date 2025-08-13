library(tidyr)
library(dplyr)
library(stringr)
library(readr)


gold <- read.csv("gold_lftk.csv")

# ----- 1.) Zip AlignScore up with Gold
AS_gold <- read_file("align_score/AlignScore=gold.txt")
align_score_list <- str_extract_all(AS_gold,"<ALIGNSCORE>.+</ALIGNSCORE>")
align_score <- c()
for(line in align_score_list){
  score <- as.numeric(str_extract(line,"<ALIGNSCORE>(.+)</ALIGNSCORE>",group = 1))
  align_score <- c(align_score,score)
}

# Pad AlignScore column since it only had 100 rows
align_score <- c(align_score,rep(NA,nrow(gold)-length(align_score)))

gold <- gold %>% cbind(align_score)


# ----- 2.) Zip SummaC up with Gold
SC_gold <- read.csv("deliverable_4/summac/gold_summac.csv")
summac <- c()
for(scoreline in SC_gold$gold_score){
  summac <- c(summac,str_extract(scoreline,"\\[(.+)\\]",group=1))
}

summac <- c(summac,rep(NA,nrow(gold)-length(summac)))

gold <- gold %>% cbind(summac)



# write new gold reference metrics file
# write.csv(gold,file = "gold_reference_metrics.csv",row.names = F)
