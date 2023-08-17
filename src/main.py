
# IMPORT GENERAL FUNCTIONS

# utility 
import json
import time
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

# model 
import argparse
import torch

# output analysis 
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression

# explainability
# import explainability

# IMPORT USER-DEFINED FUNCTIONS
from data import DataFetch, GraphData, get_batch_and_loader
from model import GNN
from utils import EarlyStopping, WeightedBCELoss, get_classification_threshold_auc, get_classification_threshold_precision_recall, brier_skill_score, calculate_metrics, enable_dropout, plot_losses

# DEFINE EXTRA FUNCTIONS

def get_model_output(model, data_batch, params):
    """Take an initialized model and return its ouput layer

    Args:
        model: initialized model object to use
        data_batch (torch_geometric.data.Dataset): data batch to use
        params: user requests, loaded using argparser

    Returns:
        model_output: dictionary containing output layers
        y: true output
    """
    
    x_static_graph          = data_batch.x.to(params['device'])
    y                       = data_batch.y.unsqueeze(1).to(params['device'])
    edge_index              = data_batch.edge_index.to(params['device'])
    edge_weight             = data_batch.edge_attr.to(params['device'])
    batch                   = data_batch.batch.to(params['device'])
    target_index            = data_batch.target_index.to(params['device'])
    
    output = model(x_static_graph, edge_index, edge_weight, batch, target_index)
    model_output = {'output':output}

    return model_output, y


activation = {}
def get_activation(name):
    """Used to get representations learned from intermediate layers
    """
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def train_model(model, train_loader, validate_loader, params):
    """ Train the desired model

    Args:
        model (): model object to use
        train_loader (): 
        validate_loader ():
        params: user requests, loaded using argparser 

    Returns:
        model_output (): 
        y (): 
    """

    model.to(params['device'])
    model_path = '{}/checkpoint_{}.pt'.format(params['outpath'], params['outname'])
    early_stopping = EarlyStopping(patience=params['patience'], path=model_path)

    if params['loss']=='bce_weighted_single' or params['loss']=='bce_weighted_sum':
        print("Using BCE weighted loss")
        train_criterion = WeightedBCELoss(params['num_samples_train_dataset'], params['num_samples_train_minority_class'], params['num_samples_train_majority_class'], params['device'])
        valid_criterion = WeightedBCELoss(params['num_samples_valid_dataset'], params['num_samples_valid_minority_class'], params['num_samples_valid_majority_class'], params['device'])
    elif params['loss']=='mse':
        train_criterion = torch.nn.MSELoss(reduction='sum')      
        valid_criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_losses = []
    valid_losses = []
    separate_loss_terms = {'NN_train':[], 'NN_valid':[]}
    
    # store for calculating classification threshold on last epoch
    valid_output = np.array([])
    valid_y = np.array([])

    #-----------------------------------------------------
    # START TRAINING
    for epoch in range(params['max_epochs']):

        separate_loss_terms_epoch = {'NN_train':[],'NN_valid':[]}    

        # evaluate model on train set
        model.train()
        epoch_train_loss = []   
        for train_batch in tqdm(train_loader, total=params['num_batches_train']):
            output, y = get_model_output(model, train_batch, params)

            if params['loss']=='bce_weighted_sum':
                # combined loss that considers the additive effect of patient and family effects
                loss_term_NN     = params['gamma'] * train_criterion(output['output'], y) 
                separate_loss_terms_epoch['NN_train'].append(loss_term_NN.item()) 
                loss = loss_term_NN 
            else:
                loss = train_criterion(output['output'], y) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        
        # evaluate on validation set
        model.eval()
        epoch_valid_loss = []
        for validate_batch in tqdm(validate_loader, total=params['num_batches_validate']):
            output, y = get_model_output(model, validate_batch, params)
            valid_output = np.concatenate((valid_output, output['output'].reshape(-1).detach().cpu().numpy()))
            valid_y = np.concatenate((valid_y, y.reshape(-1).detach().cpu().numpy()))

            if params['loss']=='bce_weighted_sum':
                # combined loss that considers the additive effect of patient and family effects
                loss_term_NN = params['gamma'] * valid_criterion(output['output'], y) 
                separate_loss_terms_epoch['NN_valid'].append(loss_term_NN.item())
                loss = loss_term_NN 
            else:
                loss = valid_criterion(output['output'], y) 

            epoch_valid_loss.append(loss.item())

        early_stopping(np.mean(epoch_valid_loss), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        train_losses.append(np.mean(epoch_train_loss))
        valid_losses.append(np.mean(epoch_valid_loss))
        for term_name in separate_loss_terms:
            separate_loss_terms[term_name].append(np.mean(separate_loss_terms_epoch[term_name]))
        print("epoch {}\ttrain loss : {}\tvalidate loss : {}".format(epoch, np.mean(epoch_train_loss), np.mean(epoch_valid_loss)))

    # STOP TRAINING
    #-----------------------------------------------------

    # load the checkpoint with the best model
    model.load_state_dict(torch.load(model_path))

    # use last values from validation set
    if params['threshold_opt'] == 'auc' & params['loss']!='mse': 
        threshold = get_classification_threshold_auc(valid_output, valid_y)
    elif params['threshold_opt'] == 'precision_recall' & params['loss']!='mse':
        threshold = get_classification_threshold_precision_recall(valid_output, valid_y)
    elif params['loss']=='mse':
        threshold = None
    
    return model, threshold

def test_model(model, test_loader, threshold, params, embeddings=False):
    num_samples = 3 # number of MC samples
    if embeddings: num_samples = 1
    test_output = [np.array([]) for _ in range(num_samples)]
    test_y = [np.array([]) for _ in range(num_samples)]

    representations = pd.DataFrame()

    model.eval()
    enable_dropout(model)
    for sample in range(num_samples):
        counter = 0
        for test_batch in tqdm(test_loader, total=params['num_batches_test']):
            output, y = get_model_output(model, test_batch, params)
            test_output[sample] = np.concatenate((test_output[sample], output['output'].reshape(-1).detach().cpu().numpy()))
            test_y[sample] = np.concatenate((test_y[sample], y.reshape(-1).detach().cpu().numpy()))
            counter += 1

    # report standard error for uncertainty
    test_output_se = np.array(test_output).std(axis=0) / np.sqrt(num_samples)

    # take average over all samples to get expected value
    test_output = np.array(test_output).mean(axis=0)
    test_y = np.array(test_y).mean(axis=0)
    results = pd.DataFrame({'actual':test_y, 'pred_raw':test_output, 'pred_raw_se':test_output_se})

    mse = metrics.mean_squared_error(test_y, test_output)
    r2 = metrics.r2_score(test_y, test_output)
    metric_results = {'MSE':mse,'R2':r2}

    return results, metric_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # the following should be updated for each experiment
    parser.add_argument('--featfile', type=str, help='filepath for featfile csv')
    parser.add_argument('--model_type', type=str, help='one of baseline, graph, graph_no_target or explainability')
    parser.add_argument('--experiment', type=str, help='a unique name for the experiment used in output file prefixes')
    parser.add_argument('--batchsize', type=int, help='batchsize for training, recommend 500 for baselines and 250 for graphs', default=500)
    
    # the following can optionally be configured for each experiment
    parser.add_argument('--outpath', type=str, help='directory for results output', default='results')
    parser.add_argument('--statfile', type=str, help='filepath for statfile csv', default='data/statfile.csv')
    parser.add_argument('--mask_target', type=str, help='mask target patient info', default='data/statfile.csv')
    parser.add_argument('--maskfile', type=str, help='filepath for maskfile csv', default='True')
    parser.add_argument('--edgefile', type=str, help='filepath for edgefile csv', default='data/edgefile.csv')
    parser.add_argument('--gnn_layer', type=str, help='type of gnn layer to use: gcn, graphconv, gat', default='graphconv')
    parser.add_argument('--use_edge', type=str, help='use or not edges in graph', default=True)
    parser.add_argument('--pooling_method', type=str, help='type of gnn pooling method to use: target, sum, mean, topkpool_sum, topkpool_mean, sagpool_sum, sagpool_mean', default='target')
    parser.add_argument('--num_workers', type=int, help='number of workers for data loaders', default=6)
    parser.add_argument('--max_epochs', type=int, help='maximum number of training epochs if early stopping criteria not met', default=100)
    parser.add_argument('--patience', type=int, help='how many epochs to wait for early stopping after last time validation loss improved', default=8)
    parser.add_argument('--learning_rate', type=float, help='learning rate for model training', default=0.001)
    parser.add_argument('--hidden_dim', type=int, help='number of hidden dimensions in (non-LSTM) neural network layers', default=20)
    parser.add_argument('--loss', type=str, help='which loss function to use: bce_weighted_single, bce_weighted_sum', default='bce_weighted_single')
    parser.add_argument('--gamma', type=float, help='weight parameter on the overall NN loss (required for bce_weighted_sum loss)', default=1)
    parser.add_argument('--alpha', type=float, help='weight parameter on the target term of the loss (required for bce_weighted_sum loss)', default=1)
    parser.add_argument('--beta', type=float, help='weight parameter on the family term of the loss (required for bce_weighted_sum loss)', default=1)
    parser.add_argument('--delta', type=float, help='weight parameter on the lstm term of the loss (required for bce_weighted_sum loss, longitudinal models only)', default=1)
    parser.add_argument('--dropout_rate', type=float, help='the dropout rate in the neural networks', default=0.5)
    parser.add_argument('--threshold_opt', type=str, help='what metric to optimize when determining the classification threshold (either auc or precision_recall)', default='precision_recall')
    parser.add_argument('--ratio', type=float, help='the graph pooling ratio for node reduction methods, determining portion of nodes to retain', default=0.5)
    
    # extra parameters used for experiments presented in paper - in general these can be ignored
    parser.add_argument('--num_positive_samples', type=int, help='number of case samples from test set used in explainability analysis', default=5000)
    parser.add_argument('--explainability_mode', action='store_true', help='explainability flag for running the post-training analysis')
    parser.add_argument('--embeddings_mode', action='store_true', help='extract the representations learned by the GNN')
    parser.add_argument('--explainer_input', type=str, help='optional explainability input file')
    parser.add_argument('--device', type=str, help='specific device to use, e.g. cuda:1, if not given detects gpu or cpu automatically', default='na')

    args = vars(parser.parse_args())

    filepaths = {'maskfile':args['maskfile'],
                'featfile':args['featfile'],
                'statfile':args['statfile'], 
                'edgefile':args['edgefile']}
    params = {'model_type':args['model_type'],
            'mask_target':args['mask_target'],
            'use_edge':args['use_edge'],
            'gnn_layer':args['gnn_layer'],
            'pooling_method':args['pooling_method'],
            'outpath':args['outpath'],
            'outname':args['experiment'],
            'batchsize':args['batchsize'], 
            'num_workers':args['num_workers'],
            'max_epochs':args['max_epochs'],
            'patience':args['patience'],
            'learning_rate':args['learning_rate'],
            'hidden_dim':args['hidden_dim'],
            'loss':args['loss'], 
            'gamma':args['gamma'], 
            'alpha':args['alpha'], 
            'beta':args['beta'], 
            'delta':args['delta'],
            'dropout_rate':args['dropout_rate'], 
            'threshold_opt':args['threshold_opt'], 
            'ratio':args['ratio'], 
            'explainability_mode':args['explainability_mode'], 
            'embeddings_mode':args['embeddings_mode'], 
            'explainer_input':args['explainer_input'],
            'device_specification':args['device'],
            'num_positive_samples':args['num_positive_samples']}
    
    if params['device_specification'] != 'na':
        params['device'] = torch.device(params['device_specification'])
    else:
        params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('STARTING DATA FETCH')
    fetch_data = DataFetch(
        maskfile=filepaths['maskfile'], 
        featfile=filepaths['featfile'], 
        statfile=filepaths['statfile'], 
        edgefile=filepaths['edgefile'], 
        params=params,)
    
    train_patient_list = fetch_data.train_patient_list
    params['num_batches_train'] = int(np.ceil(len(train_patient_list)/params['batchsize']))
    params['num_samples_train_dataset'] = len(fetch_data.train_patient_list)
    params['num_samples_train_minority_class'] = fetch_data.num_samples_train_minority_class
    params['num_samples_train_majority_class'] = fetch_data.num_samples_train_majority_class

    validate_patient_list = fetch_data.validate_patient_list
    params['num_batches_validate'] = int(np.ceil(len(validate_patient_list)/params['batchsize']))
    params['num_samples_valid_dataset'] = len(fetch_data.validate_patient_list)
    params['num_samples_valid_minority_class'] = fetch_data.num_samples_valid_minority_class
    params['num_samples_valid_majority_class'] = fetch_data.num_samples_valid_majority_class

    test_patient_list = fetch_data.test_patient_list
    params['num_batches_test'] = int(np.ceil(len(test_patient_list)/params['batchsize']))

    print('STARTING BATCH PREPARATION')
    train_dataset, train_loader = get_batch_and_loader(train_patient_list, fetch_data, params, shuffle=True)
    validate_dataset, validate_loader = get_batch_and_loader(validate_patient_list, fetch_data, params, shuffle=True)
    test_dataset, test_loader = get_batch_and_loader(test_patient_list, fetch_data, params, shuffle=False)
    params['num_features_static'] = len(fetch_data.static_features)
    
    model = GNN(
        num_features_static_graph   = params['num_features_static'], 
        hidden_dim                  = params['hidden_dim'], 
        gnn_layer                   = params['gnn_layer'], 
        pooling_method              = params['pooling_method'], 
        dropout_rate                = params['dropout_rate'], 
        ratio                       = params['ratio'])

    model_path = '{}/{}_model.pth'.format(params['outpath'], params['outname'])
    results_path = '{}/{}_results.csv'.format(params['outpath'], params['outname'])
    stats_path = '{}/{}_stats.csv'.format(params['outpath'], params['outname'])

    print('STARTING MODEL TRAIN/TEST')  

    if params['explainability_mode']:
        results = pd.read_csv(results_path)
        # select graphs to explain
        samples = explainability.sampling(results, num_positive_samples=params['num_positive_samples'], uncertainty_rate=0.9)
        exp_patient_list = test_patient_list[samples]
        # load one graph at a time
        params['batchsize'] = 1
        exp_dataset, exp_loader = get_batch_and_loader(exp_patient_list, fetch_data, params, shuffle=False)

        # free up memory no longer needed
        del fetch_data 
        del train_dataset
        del validate_dataset
        del test_dataset

        print("Loading model")
        model.load_state_dict(torch.load(model_path))
        model.to(params['device'])
        torch.backends.cudnn.enabled = False
        explainability.gnn_explainer(model, exp_loader, exp_patient_list, params)
    
    elif params['embeddings_mode']:
        # use same samples used for explainability
        exp_data = pd.read_csv(params['explainer_input'])
        exp_patient_list = torch.tensor([int(e) for e in list(exp_data['target_id'].unique())])
        exp_dataset, exp_loader = get_batch_and_loader(exp_patient_list, fetch_data, params, shuffle=False)
        params['num_batches_test'] = int(np.ceil(len(exp_patient_list)/params['batchsize']))
        
        # free up memory no longer needed
        del fetch_data 
        del train_dataset
        del validate_dataset
        del test_dataset
        del exp_data

        print("Loading model")
        model.load_state_dict(torch.load(model_path))
        model.combined_conv2.register_forward_hook(get_activation('combined_conv2'))
        model.to(params['device'])

        stats = pd.read_csv(stats_path)
        stats_dict = dict(zip(stats['name'],stats['value']))
        threshold = float(stats_dict['threshold'])
        results, metric_results = test_model(model, exp_loader, threshold, params, embeddings=True)
        
    else:
        # normal training model
        # free up memory no longer needed
        del fetch_data 
        del train_dataset
        del validate_dataset
        del test_dataset

        # model training
        start_time_train = time.time()
        model, threshold = train_model(model, train_loader, validate_loader, params)
        end_time_train = time.time()
        torch.save(model.state_dict(), model_path)
        params['threshold'] = threshold
        params['training_time'] = end_time_train - start_time_train

        # model testing
        results, metric_results = test_model(model, test_loader, threshold, params)
        results.to_csv(results_path, index=None)
        params.update(metric_results)
        stats = pd.DataFrame({'name':list(params.keys()), 'value':list(params.values())})
        stats.to_csv(stats_path, index=None)

