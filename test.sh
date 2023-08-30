source /data/projects/project_GNN/envs/graphml/bin/activate

outpath=output 
experiment=test_0 

statfile=/data/projects/project_GNN/gnn_family_pedigree/data/statfile.csv
maskfile=/data/projects/project_GNN/gnn_family_pedigree/data/maskfile.csv
edgefile=/data/projects/project_GNN/gnn_family_pedigree/data/edgefile.csv
featfile=/data/projects/project_GNN/gnn_family_pedigree/data/featfile.csv 

model_type=graph
gnn_layer=gcn
use_edge=False
mask_target=True
pooling_method=mean
num_workers=8
max_epochs=25
patience=5
learning_rate=0.001
hidden_dim=20
dropout_rate=0.5
threshold_opt=auc
loss=mse
gamma=1
alpha=1
beta=1
delta=1
batchsize=250

device='cuda:1' 

nohup python3 ./src/main.py --featfile ${featfile} --model_type ${model_type} --experiment ${experiment} --batchsize ${batchsize} --outpath ${outpath} --statfile ${statfile} --maskfile ${maskfile} --edgefile ${edgefile} --mask_target ${mask_target} --gnn_layer ${gnn_layer} --use_edge ${use_edge} --pooling_method ${pooling_method} --num_workers ${num_workers} --max_epochs ${max_epochs} --patience ${patience} --learning_rate ${learning_rate} --hidden_dim ${hidden_dim} --loss ${loss} --gamma ${gamma} --alpha ${alpha} --beta ${beta} --delta ${delta} --dropout_rate ${dropout_rate} --threshold_opt ${threshold_opt} --device ${device} > ${outpath}/${experiment}.txt 
