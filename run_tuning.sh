source /data/projects/project_GNN/gnn_family_pedigree/env/bin/activate

outpath=output/results_tuning_0
experiment=chd_baseline_undir

statfile=/data/projects/project_GNN/gnn_family_pedigree/data/statfile.csv
maskfile=/data/projects/project_GNN/gnn_family_pedigree/data/maskfile.csv
edgefile=/data/projects/project_GNN/gnn_family_pedigree/data/edgefile_onlyparents.csv
featfile=/data/projects/project_GNN/gnn_family_pedigree/data/featfile_chd.csv 

model_type=tuning

gnn_layer=gcn
use_edge=False
mask_target=True
directed=False
pooling_method=sum
num_workers=20
patience=20
max_epochs=100
learning_rate=0.001
hidden_dim=64
hidden_layers=1
dropout_rate=0.4
threshold_opt=auc
loss=weighted_bce
gamma=1
alpha=1
beta=1
delta=1
batchsize=500
device='cuda:1' 

nohup python3 ./src/main.py --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --edgefile ${edgefile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --tuning_mode > ${outpath}/${experiment}_tuning.txt 