
# to run on ePouta:
# conda activate /data/tools/conda
# python <filename.py>

#-------------------------------------------------------------------------------
# PREPARE ENVIRONMENT
import pandas as pd
import numpy as np
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split

START = datetime.now()
MIN_PHENO_PATH = "/data/processed_data/minimal_phenotype/minimal_phenotype_2023-08-14.csv"
RELATIVES_PATH = "/data/projects/project_SophieAndZhiyu/Relatives/family_relationships.csv"
MASKFILE_PATH = "/data/projects/project_GNN/GAT_family_pedigree/data/maskfile.csv"
STATFILE_PATH = "/data/projects/project_GNN/GAT_family_pedigree/data/statfile.csv"
EXTRA_FEATURES = "/data/projects/project_GNN/AllIndFeat"
# using risteys endpoint
ENDPOINT_NAME = 'I9_CHD_nEvent'

# FETCH DATA
df = pd.read_csv(MIN_PHENO_PATH, sep = ',', usecols=["FINREGISTRYID","INDEX_PERSON","DATE_OF_BIRTH","SEX","MOTHER_TONGUE","EMIGRATED"])
df = df.rename(str.lower,axis='columns').rename(columns={'finregistryid':'FINREGISTRYID'})
relatives = pd.read_csv(RELATIVES_PATH, sep = ',')

#-------------------------------------------------------------------------------
# DEFINE FUNCTIONS
def apply_exclusion_criteria(df):
    # check that required columns are present
    N_START = df.shape[0]
    assert N_START>0, 'empty population dataframe'
    print(f'starting from {N_START} patients available in Finregistry')
    assert 'date_of_birth' in df, 'missing required columns for exclusion criterias'
    assert 'sex' in df, 'missing required columns for exclusion criterias'
  
    df['birth_year'] = df.date_of_birth.str[:4].astype('int64')
    df = df[~df.date_of_birth.isna()]
    print(f'{df.shape[0]} patients with NOT missing date of birth')
    df = df[~df.sex.isna()]
    print(f'{df.shape[0]} patients with NOT missing sex')
    df = df[~df.birth_year<1920]
    print(f'{df.shape[0]} patients born after 1920')
    print(f'after exclusion criteria, {df.shape[0]} patients are enetring the study')
    return df

def define_target_patients(df, relatives):
    #check that required columns are present
    assert set(['index_person','mother_tongue','emigrated']).issubset(df.columns), 'missing required columns for definition of target patient'
    
    # remove non eligible (excluded) people
    eligible = df.FINREGISTRYID.tolist()
    relatives = relatives[ (relatives.ID1.isin(eligible)) & (relatives.ID2.isin(eligible)) ] 
    
    # check parental connections
    has_mother      = set(relatives[(relatives.relationship == 'parent') & (relatives.relationship_detail == 'mother')].ID1.tolist())
    has_father      = set(relatives[(relatives.relationship == 'parent') & (relatives.relationship_detail == 'father')].ID1.tolist())
    has_grandparent = set(relatives[relatives.relationship == 'grandparent'].ID1.tolist())
    has_required_connections = list(set.intersection(has_mother, has_father, has_grandparent))
    print(f'{len(has_required_connections)} have both parents and at least 1 grandparent')
    
    df["target"] = 0
    # define one single rules with AND conditions
    is_target = ( 
      (df.index_person==1) & 
      (df.mother_tongue.isin(['fi','sv'])) & 
      ((df.emigrated == 0) | (df.emigrated.isna())) &
      ((df.birth_year>=1970) & (df.birth_year<=1990)) &
      (df.FINREGISTRYID.isin(has_required_connections))
    )
    df.loc[is_target,"target"] = 1
    print(f'{sum(is_target)} patients are going to be used as target patients')
    return df
  
def train_test_split(df):
    # 70% train, 10% valid, 20% test 
    train_valid_test_split = [0.7,0.1,0.2]
    # -1=not target, 0=train, 1=valid , 2=test
    df["train"] = -1
    N_TARGETS = sum(df.target==1)
    df.loc[df.target==1,"train"] = np.random.choice([0,1,2], N_TARGETS, p=train_valid_test_split)
    
    print(f'there are {sum(df.train==0)} training target patients')
    print(f'there are {sum(df.train==1)} validation target patients')
    print(f'there are {sum(df.train==2)} test target patients')
    return df
 

#-------------------------------------------------------------------------------
# EXTRACT STUDY POPULATION 

df = apply_exclusion_criteria(df)
df['node_id'] = np.arange(df.shape[0]).astype('int64')
df = define_target_patients(df,relatives)
df = train_test_split(df)
df = df.sort_values(by=['node_id'])

# create maskfile and export
MASKFILE_COLS = ['FINREGISTRYID','node_id','target','train']
df[MASKFILE_COLS].to_csv(MASKFILE_PATH , index=False)

# add info about the disease of interest, then save to statfile
extra = pd.read_csv(EXTRA_FEATURES, sep=',',usecols=['FINREGISTRYID',ENDPOINT_NAME]) 
df = df.merge(extra, how='left',on='FINREGISTRYID')
df[ENDPOINT_NAME+'_binary'] = (df[ENDPOINT_NAME]>0).astype('int64')
df.to_csv(STATFILE_PATH,index=False)

print(f'finished in {datetime.now()-START} hr:min:sec')
gc.collect()
