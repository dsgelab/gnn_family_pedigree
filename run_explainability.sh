source /data/projects/project_GNN/gnn_family_pedigree/env/bin/activate

outpath=output/results_test_4
experiment=chd_EndPt_undir

maskfile=/data/projects/project_GNN/gnn_family_pedigree/data/maskfile.csv
edgefile=/data/projects/project_GNN/gnn_family_pedigree/data/edgefile_onlyparents.csv
statfile=/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/statfile_all.csv
featfile=/data/projects/project_GNN/gnn_family_pedigree/data/extended_corr02/featfile_chd_all.csv 

model_type=explainability
num_positive_samples=1000

gnn_layer=gcn
use_edge=False
mask_target=True
pooling_method=target
num_workers=20
patience=10
max_epochs=50
learning_rate=0.001
hidden_dim=64
dropout_rate=0.2
threshold_opt=auc
loss=weighted_bce
gamma=1
alpha=1
beta=1
delta=1
batchsize=500
device='cuda:1' 

nohup python3 ./src/main.py --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --edgefile ${edgefile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} --num_positive_samples ${num_positive_samples} --explainability_mode > ${outpath}/${experiment}_explainability.txt 