
library(data.table)
library(dplyr)

# needs edgefile and statfile
PROJECT_PATH = "/data/projects/project_GNN/age_prediction/"
STAT_PATH = paste0(PROJECT_PATH,"data/statfile.csv")
EDGE_PATH = paste0(PROJECT_PATH,"data/edgefile.csv")

# fetch data
statfile = fread(STAT_PATH)
edgefile = fread(EDGE_PATH)

# define target patients
tmp = edgefile %>% 
  filter(relationship_type=='parent_na') %>%
  group_by(node1) %>%
  summarise(N_PARENTS=n()) %>%
  ungroup() %>%
  rename('node_id'='node1') %>%
  full_join(statfile,by = 'node_id') %>%
  mutate(target = if_else(N_PARENTS==2, true=1, false=0, missing=0)) 

# apply exclusion criteria
tmp[tmp$index_person != 1, 'target'] = 0
tmp[tmp$emigrated != 0 | is.na(tmp$emigrated), 'target'] = 0
tmp[!(tmp$mother_tongue %in% c('fi','sv')), 'target'] = 0

# GC: sum(tmp$target == 1)
# there are 3.682.845 target patients

# define training set 
set.seed(1)
tmp[tmp$target == 0, 'train'] = -1
tmp[tmp$target == 1, 'train'] = sample(x = c(0,1,2), 
                                       size = sum(tmp$target==1), 
                                       replace = TRUE, 
                                       prob = c(0.7,0.1,0.2))

# QC: tmp %>% group_by(train) %>% summarise(TOT = n())

# save results (and update other files)
maskfile = tmp[, c('FINREGISTRYID','graph','target','node_id','train')] %>% arrange(FINREGISTRYID)
statfile = tmp %>% select(-c('N_PARENTS')) %>% arrange(FINREGISTRYID)
edgefile$target =  tmp$target[match(edgefile$node1,tmp$node_id)]

fwrite(maskfile, paste0(PROJECT_PATH,"data/maskfile.csv"), row.names = F)
fwrite(statfile, paste0(PROJECT_PATH,"data/statfile.csv"), row.names = F)
fwrite(edgefile, paste0(PROJECT_PATH,"data/edgefile.csv"), row.names = F)

rm(list=ls())
gc()

