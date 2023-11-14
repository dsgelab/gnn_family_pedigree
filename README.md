
# GNN ON FAMILY PEDIGREE

Repository containing the code used for evaluating a Graph Neural Network (GNN) model for family pedigree data.

RESEARCH QUESTION: <br>
Can we “impute” a phenotype by knowing nothing about the target individual and only leveraging information for each node in the familial pedigree?

## FILES

### FILE STRUCTURE
```
|--- / 

    pipeline_data.sh 
    pipeline_model.sh 
    run_explainability.sh 

    |--- data/
        statfile.csv
        maskfile.csv
        edgefile_onlyparents.csv
        featfile_chd.csv
        |--- extended_data/
            statfile_Drug.csv
            statfile_EndPt.csv
            statfile_SES.csv
            statfile_all.csv
            featfile_chd_Drug.csv
            featfile_chd_EndPt.csv
            featfile_chd_SES.csv
            featfile_chd_all.csv
        |--- scripts/
            extract_study_population.py
            extract_edge_onlyparents.py
            add_extra_features.py

    |--- src/
        main.py
        data.py
        model.py
        utils.py
        explainability.py
        my_explainability.py

    |--- logs/
    |--- output/
```

### FILE CONTENT

**data/** <br>
the folder contains all input files used for the GNN models:
- _statfile_ &nbsp;&nbsp;&emsp;: contains basic information available for every patient 
- _maskfile_ &emsp;: specifies if patient is used in the project, if it is a target patient and if it is used to train/validate or test the model
- _edgefile_ &nbsp;&emsp;: specifies all the graph edges i.e. all the connections between patients
- _featfile_ &nbsp;&nbsp;&emsp;: needs to be manually generated, specifies the features to be used for training the model

plus the scripts used to create them:
- _extract_study_population.py_ &nbsp;&emsp;: create statfile.csv and maskfile.csv
- _extract_edge_onlyparents.py_ &emsp;: create edgefile_onlyparents.csv
- _add_extra_features.py_ &emsp;&emsp;&emsp;&emsp;: extend the main statfile with extra registry information (see extended_data folder)

NB: <br>
**extended_data/** can be substituted with another folder containing a different extension of the stafiles, e.g. using a subsample of all the available covariates

**src/** <br>
the folder contains all the scripts used for the GNN:
- _utils.py_ &nbsp;&emsp;: utility functions
- _model.py_ &nbsp;&nbsp;: GNN model architecture 
- _data.py_  &nbsp;&emsp;: construct pytorch_geometric objects 
- _main.py_  &emsp;: perform model train and test 

plus the shell pipelines used:
- pipeline_data.sh &emsp;&emsp;&emsp;: used for extracting the study population and create the GNN input files
- pipeline_models.sh &emsp;&nbsp;&nbsp;&nbsp;: used for training and testing the desired models
- run_explainability.sh &emsp;: used for extracting the GNNExpaliner results on the desired model


# REFERENCES

project inspired by Sophie Wharrie's paper on a similar analysis in finregistry <br>
PREPRINT: 
https://arxiv.org/abs/2304.05010 



# PEOPLE

**CODE AUTHOR** 
- Matteo Ferro  &nbsp;&emsp;&emsp;matteo.ferro@heslinki.fi

**COLLABORATORS**
- Zhiyu Yang &nbsp;&nbsp;&emsp;&emsp;&emsp;zhiyu.yang@helsinki.fi
- Sophie Wharrie  &nbsp;&nbsp;&emsp;sophie.wharrie@aalto.fi
