source /data/projects/project_GNN/GAT_family_pedigree/env/bin/activate

statfile=/data/projects/project_GNN/GAT_family_pedigree/data/statfile_test.csv
maskfile=/data/projects/project_GNN/GAT_family_pedigree/data/maskfile_test.csv
featfile=/data/projects/project_GNN/GAT_family_pedigree/data/featfile_chd.csv 

gnn_layer=gat
use_edge=False
mask_target=True
directed=True
add_self_loops=False
pooling_method=target
num_workers=20
patience=20
max_epochs=100
learning_rate=0.01
hidden_dim=64
hidden_layers=3
dropout_rate=0.1
ratio=0.5
threshold_opt=auc
loss=weighted_bce
gamma=1
alpha=1
beta=1
delta=1
batchsize=10
device='cuda:1' 

model_type=tuning
outpath=output/

aggr_func=mean
experiment=chd_GAT_mean
nohup python3 ./src/main.py --aggr_func ${aggr_func} --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --tuning_mode > ${outpath}/${experiment}_tuning.txt 

aggr_func=min
experiment=chd_GAT_min
nohup python3 ./src/main.py --aggr_func ${aggr_func} --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --tuning_mode > ${outpath}/${experiment}_tuning.txt 

aggr_func=max
experiment=chd_GAT_max
nohup python3 ./src/main.py --aggr_func ${aggr_func} --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --tuning_mode > ${outpath}/${experiment}_tuning.txt 
