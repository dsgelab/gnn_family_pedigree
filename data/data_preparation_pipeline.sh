
echo "extract study population ... create statfile"
/opt/R-4.1.1/bin/Rscript create_statfile.R &>/dev/null

echo "extract pedigree info ... create edgefile"
source /data/projects/project_GNN/envs/graphml/bin/activate
python create_edgefile.py 
source /data/projects/project_GNN/envs/graphml/bin/deactivate

echo "extract model trainig info ... create maskfile"
/opt/R-4.1.1/bin/Rscript create_maskfile.R &>/dev/null

echo "FINISHED .. here are the results"
head -5 statfile.csv
head -5 edgefile.csv
head -5 maskfile.csv


