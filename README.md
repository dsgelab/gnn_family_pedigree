
# PROJECT 

to run the project GNN with desired prameters use _test.sh_ file. <br>
the result can be found in the folder **output/**

# FILES
**data/** folder contains all the data used in the project and the scripts to create them, which are:
- _extract_study_population.py_ 	: create statfile.csv and maskfile.csv
- _extract_edge_info.py_        	: create edgefile.csv
- _extract_bothparent_patients.py_	: create both_parents_list.txt

STATFILE : contains all the feature information for every patient available

MASKFILE : specify if patient is used in the project, if it is a target and if it is used to train/validate or test the model

EDGEFILE : specify all the graph edges i.e. all the connections between patients

**src/** folder contains all the scripts to run the project, which means:
- _utils.py_ : utility functions
- _model.py_ : GNN model architecture 
- _data.py_  : construct pytorch_geometric objects 
- _main.py_  : perform model train and test

