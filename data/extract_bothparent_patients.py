
# PREPARE ENVIRONMENT

# run this in ePouta: 
# source /data/projects/project_GNN/envs/graphml/bin/activate

import pandas as pd

RELATIVES_PATH = "/data/projects/project_SophieAndZhiyu/Relatives/family_relationships.wMarrigeChild"
df = pd.read_csv(RELATIVES_PATH)
occurrences = df[df.relationship == 'parent'].groupby('ID1').size()
print(f'starting with {len(occurrences)} patients')
out = occurrences[occurrences==2]
print(f'{len(out)} patients with both parents')
registry_ids = out.index.tolist()
print('saving list of patient into file: both_parents_list.txt ')
with open(r'/data/projects/project_GNN/gnn_family_pedigree/data/both_parents_list.txt', 'w') as fp:
    fp.write('\n'.join(registry_ids))
    
