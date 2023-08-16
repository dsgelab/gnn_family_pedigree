
library(data.table)
library(dplyr)

# starting from minimal_phenotype in Finregistry extract the study population

# EXCLUSION CRITERIA: 
#   non index person
#   emigrated
#   non-finnish speaker
#   younger then 18 at chosen time-point

# COVARIATES OF INTEREST
#   sex
#   age at chosen time point
#   is alive at chosen time point

MIN_PHENO_PATH = "/data/processed_data/minimal_phenotype/archive/minimal_phenotype_2022-03-28.csv"
PROJECT_PATH = "/data/projects/project_GNN/age_prediction/"
TIME_POINT = "2010-01-01"

# prepare data
df = fread(MIN_PHENO_PATH, sep = ',')
df = df %>% mutate(
  age   = as.numeric(as.difftime(as.Date(TIME_POINT) - as.Date(df$date_of_birth), units = 'days'))/365.25,
  alive = if_else((as.Date(TIME_POINT)<as.Date(df$death_date)) | (is.na(df$death_date)),1,0), graph = 1) %>%
  arrange(FINREGISTRYID)

# apply exclusion criteria
# starting with 7.166.416 patients
df[is.na(df$date_of_birth), 'graph'] = 0
df[is.na(df$sex), 'graph'] = 0
# QC: sum(df$graph == 1)
# ending with 7.070.388 patients

# define node ID
df[df$graph==1,'node_id'] = 1:sum(df$graph==1)

# save results
statfile = df[, c('FINREGISTRYID','age','sex','alive','mother_tongue','emigrated','index_person','graph','node_id')]
fwrite(statfile, paste0(PROJECT_PATH,"data/statfile.csv"), row.names = F)

rm(list=ls())
gc()

