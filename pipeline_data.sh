
# activate virtual environment
source /data/projects/project_GNN/envs/graphml/bin/activate

python data/scripts/extract_study_population.py > logs/log_study_pop.txt
python data/scripts/extract_edge_onlyparents.py > logs/log_edges.txt
python data/scripts/add_extra_features.py > logs/log_add_features.txt
