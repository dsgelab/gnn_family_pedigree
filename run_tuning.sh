source /data/projects/project_GNN/gnn_family_pedigree/env/bin/activate

statfile=/data/projects/project_GNN/gnn_family_pedigree/data/statfile.csv
maskfile=/data/projects/project_GNN/gnn_family_pedigree/data/maskfile.csv
edgefile=/data/projects/project_GNN/gnn_family_pedigree/data/edgefile_onlyparents.csv
featfile=/data/projects/project_GNN/gnn_family_pedigree/data/featfile_chd.csv 

gnn_layer=gcn
use_edge=False
mask_target=True
directed=False
add_self_loops=False
pooling_method=sum
num_workers=20
patience=20
max_epochs=100
learning_rate=0.001
hidden_dim=64
hidden_layers=1
dropout_rate=0.4
ratio=0.5
threshold_opt=auc
loss=weighted_bce
gamma=1
alpha=1
beta=1
delta=1
batchsize=500
device='cuda:1' 

model_type=tuning
outpath=output/results_tuning

#----------------------------
# TRAIN/TEST UNDIRECTED MODEL

experiment=chd_all_undir
statfile=/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_all.csv
featfile=/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/featfile_chd_all.csv 

nohup python3 ./src/main.py --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --edgefile ${edgefile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --tuning_mode > ${outpath}/${experiment}_tuning.txt 

#----------------------------
# TRAIN/TEST DIRECTED MODEL
directed=True

experiment=chd_all_dir
statfile=/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_all.csv
featfile=/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/featfile_chd_all.csv 

nohup python3 ./src/main.py --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --edgefile ${edgefile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --tuning_mode > ${outpath}/${experiment}_tuning.txt 
