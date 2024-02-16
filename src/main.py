
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
import optuna

# output analysis 
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression

# IMPORT USER-DEFINED FUNCTIONS
from data import DataFetch, GraphData, get_batch_and_loader
from model import GNN
from utils import EarlyStopping, WeightedBCELoss, get_classification_threshold_auc, get_classification_threshold_precision_recall, brier_skill_score, calculate_metrics, enable_dropout, plot_losses

# DEFINE EXTRA FUNCTIONS

def get_model_output(model, data_batch, params):
    
    x                       = data_batch.x.to(params['device'])
    y                       = data_batch.y.unsqueeze(1).to(params['device'])
    edge_index              = data_batch.edge_index.to(params['device'])
    batch                   = data_batch.batch.to(params['device'])
    target_index            = data_batch.target_index.to(params['device'])

    # look in forward() in model.py for model architecture
    output = model(x, edge_index, batch, target_index)
    model_output = {'output':output}

    return model_output, y


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook




# Define the objective tuning function for Optuna
def hyperparameter_tuning(trial, train_loader, validate_loader, params):
    
    num_features_static_graph = params['num_features_static']
    learning_rate = params['learning_rate']
    self_loops = params['add_self_loops']
    ratio = params['ratio']
    gnn_layer = params['gnn_layer']
    pooling_method = params['pooling_method']
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 512, step=32)
    hidden_dim_2 = trial.suggest_int('hidden_dim_2', 32, 512, step=32)
    hidden_layers = trial.suggest_int('hidden_layers', 1, 3, step=1)
    
    model = GNN(
        num_features_static_graph   = num_features_static_graph, 
        hidden_dim                  = hidden_dim,
        hidden_dim_2                = hidden_dim_2,
        hidden_layers               = hidden_layers,
        gnn_layer                   = gnn_layer, 
        pooling_method              = pooling_method, 
        dropout_rate                = dropout_rate, 
        ratio                       = ratio,
        self_loops                  = self_loops )
        
    model.to(params['device'])

    if params['loss']=='bce':
        train_criterion = torch.nn.BCEWithLogitsLoss()
        valid_criterion = torch.nn.BCEWithLogitsLoss()
    elif params['loss']=='weighted_bce':
        train_criterion = WeightedBCELoss(params['num_samples_train_dataset'], params['num_samples_train_minority_class'], params['num_samples_train_majority_class'], params['device'])
        valid_criterion = WeightedBCELoss(params['num_samples_valid_dataset'], params['num_samples_valid_minority_class'], params['num_samples_valid_majority_class'], params['device'])
    elif params['loss']=='mse':
        train_criterion = torch.nn.MSELoss(reduction='sum')      
        valid_criterion = torch.nn.MSELoss(reduction='sum')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/100)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=learning_rate*10, mode='triangular2', cycle_momentum=False)

    # evaluate model on train set
    for epoch in range(10): 
        model.train()
        for train_batch in train_loader:
            output, y = get_model_output(model, train_batch, params)
            loss = train_criterion(output['output'], y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print("completed epoch {}, with loss: {}".format(epoch,torch.round(loss,decimals=3)))
        
    # evaluate on validation set
    valid_output = np.array([])
    valid_y = np.array([])
    epoch_valid_loss = []
    model.eval()
    
    for validate_batch in validate_loader:
        output, y = get_model_output(model, validate_batch, params)
        valid_output = np.concatenate((valid_output, output['output'].reshape(-1).detach().cpu().numpy()))
        valid_y = np.concatenate((valid_y, y.reshape(-1).detach().cpu().numpy()))
        loss = valid_criterion(output['output'], y) 
        epoch_valid_loss.append(loss.item())
        
    # final result     
    val_loss = np.mean(epoch_valid_loss)
    print('learning_rate, dropout_rate, ratio, hidden_dim, hidden_dim_2, hidden_layers, gnn_layer, self_loops, pooling_method')
    print(learning_rate, dropout_rate, ratio, hidden_dim, hidden_dim_2, hidden_layers, gnn_layer, self_loops, pooling_method)

    return val_loss



def train_model(model, train_loader, validate_loader, params):

    model.to(params['device'])
    model_path = '{}/checkpoint_{}.pt'.format(params['outpath'], params['outname'])
    early_stopping = EarlyStopping(patience=params['patience'], path=model_path)

    if params['loss']=='bce':
        train_criterion = torch.nn.BCEWithLogitsLoss()
        valid_criterion = torch.nn.BCEWithLogitsLoss()
    elif params['loss']=='weighted_bce':
        train_criterion = WeightedBCELoss(params['num_samples_train_dataset'], params['num_samples_train_minority_class'], params['num_samples_train_majority_class'], params['device'])
        valid_criterion = WeightedBCELoss(params['num_samples_valid_dataset'], params['num_samples_valid_minority_class'], params['num_samples_valid_majority_class'], params['device'])
    elif params['loss']=='mse':
        train_criterion = torch.nn.MSELoss(reduction='sum')      
        valid_criterion = torch.nn.MSELoss(reduction='sum')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=(params['learning_rate']/100))
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=params['learning_rate'], max_lr=params['learning_rate']*10, mode='triangular2', cycle_momentum=False)

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
        
        for train_batch in train_loader:
            output, y = get_model_output(model, train_batch, params)
            loss = train_criterion(output['output'], y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss.append(loss.item())
            
        print("completed epoch {}".format(epoch))
        
        # evaluate on validation set
        model.eval()
        epoch_valid_loss = []
        
        for validate_batch in validate_loader:
            output, y = get_model_output(model, validate_batch, params)
            valid_output = np.concatenate((valid_output, output['output'].reshape(-1).detach().cpu().numpy()))
            valid_y = np.concatenate((valid_y, y.reshape(-1).detach().cpu().numpy()))
           
            loss = valid_criterion(output['output'], y) 
            epoch_valid_loss.append(loss.item())
        
        # check if early stopping       
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
    
    # plot_losses
    plot_losses(train_losses, valid_losses, '{}/{}'.format(params['outpath'], params['outname']))

    # use last values from validation set
    if params['threshold_opt']== 'auc' and params['loss']=='bce': 
        threshold = get_classification_threshold_auc(valid_output, valid_y)
    elif params['threshold_opt']== 'precision_recall' and params['loss']=='bce':
        threshold = get_classification_threshold_precision_recall(valid_output, valid_y)
    elif params['threshold_opt']== 'auc' and params['loss']=='weighted_bce': 
        threshold = get_classification_threshold_auc(valid_output, valid_y)
    elif params['threshold_opt']== 'precision_recall' and params['loss']=='weighted_bce':
        threshold = get_classification_threshold_precision_recall(valid_output, valid_y)    
    elif params['loss']=='mse':
        threshold = None
    
    return model, threshold

def test_model(model, test_loader, threshold, params):

    num_samples = 10 # number of MC samples
    test_output = [np.array([]) for _ in range(num_samples)]
    test_y = [np.array([]) for _ in range(num_samples)]

    representations = pd.DataFrame()
    model.eval()
    enable_dropout(model)

    for sample in range(num_samples):
        counter = 0
        for test_batch in test_loader:
            output, y = get_model_output(model, test_batch, params)
            test_output[sample] = np.concatenate((test_output[sample], output['output'].reshape(-1).detach().cpu().numpy()))
            test_y[sample] = np.concatenate((test_y[sample], y.reshape(-1).detach().cpu().numpy()))
            counter += 1

    # report standard error for uncertainty
    test_output_se = np.array(test_output).std(axis=0) / np.sqrt(num_samples)

    # take average over all samples to get expected value
    test_output = np.array(test_output).mean(axis=0)
    test_y = np.array(test_y).mean(axis=0)
    
    if (params['loss']=='bce') or (params['loss']=='weighted_bce'):
        results = pd.DataFrame({'actual':test_y, 'pred_raw':test_output, 'pred_raw_se':test_output_se})
        results['pred_binary'] = (results['pred_raw']>threshold).astype(int)
        metric_results = calculate_metrics(test_y, results['pred_binary'], test_output)
    elif params['loss']=='mse':
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
    parser.add_argument('--mask_target', type=str, help='mask target patient info', default=True)
    parser.add_argument('--maskfile', type=str, help='filepath for maskfile csv', default='True')
    parser.add_argument('--gnn_layer', type=str, help='type of gnn layer to use: gcn, graphconv, gat', default='graphconv')
    parser.add_argument('--aggr_func', type=str, help='function used to aggreagte the family clusters', default='mean')
    parser.add_argument('--use_edge', type=str, help='use or not edges in graph', default=True)
    parser.add_argument('--add_self_loops', type=str, help='use or not self loops in graph', default=False)
    parser.add_argument('--directed', type=str, help='use or not edges in graph', default=True)
    parser.add_argument('--pooling_method', type=str, help='type of gnn pooling method to use: target, sum, mean, topkpool_sum, topkpool_mean, sagpool_sum, sagpool_mean', default='target')
    parser.add_argument('--num_workers', type=int, help='number of workers for data loaders', default=6)
    parser.add_argument('--max_epochs', type=int, help='maximum number of training epochs if early stopping criteria not met', default=100)
    parser.add_argument('--patience', type=int, help='how many epochs to wait for early stopping after last time validation loss improved', default=8)
    parser.add_argument('--learning_rate', type=float, help='learning rate for model training', default=0.001)
    parser.add_argument('--hidden_dim', type=int, help='number of hidden dimensions in (non-LSTM) neural network layers', default=20)
    parser.add_argument('--hidden_dim_2', type=int, help='number of hidden dimensions in (non-LSTM) neural network layers', default=20)
    parser.add_argument('--hidden_layers', type=int, help='number of hidden layers after input layer in the network ', default=1)
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
    parser.add_argument('--tuning_mode', action='store_true', help='hyperparameter tuning flag')
    parser.add_argument('--explainer_input', type=str, help='optional explainability input file')
    parser.add_argument('--device', type=str, help='specific device to use, e.g. cuda:1, if not given detects gpu or cpu automatically', default='na')

    args = vars(parser.parse_args())

    filepaths = {'maskfile':args['maskfile'],
                'featfile':args['featfile'],
                'statfile':args['statfile']}
    params = {'model_type':args['model_type'],
            'mask_target':args['mask_target'],
            'use_edge':args['use_edge'],
            'add_self_loops':args['add_self_loops'],
            'directed':args['directed'],
            'gnn_layer':args['gnn_layer'],
            'aggr_func':args['aggr_func'],
            'pooling_method':args['pooling_method'],
            'outpath':args['outpath'],
            'outname':args['experiment'],
            'batchsize':args['batchsize'], 
            'num_workers':args['num_workers'],
            'max_epochs':args['max_epochs'],
            'patience':args['patience'],
            'learning_rate':args['learning_rate'],
            'hidden_dim':args['hidden_dim'],
            'hidden_dim_2':args['hidden_dim_2'],
            'hidden_layers':args['hidden_layers'],
            'loss':args['loss'], 
            'gamma':args['gamma'], 
            'alpha':args['alpha'], 
            'beta':args['beta'], 
            'delta':args['delta'],
            'dropout_rate':args['dropout_rate'], 
            'threshold_opt':args['threshold_opt'], 
            'ratio':args['ratio'], 
            'tuning_mode':args['tuning_mode'],
            'explainability_mode':args['explainability_mode'], 
            'explainer_input':args['explainer_input'],
            'device_specification':args['device'],
            'num_positive_samples':args['num_positive_samples']}
    
    if params['device_specification'] != 'na':
        params['device'] = torch.device(params['device_specification'])
    else:
        params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('using the following device: {}'.format(params['device']))
    print('STARTING DATA FETCH')
    fetch_data = DataFetch(
        maskfile=filepaths['maskfile'], 
        featfile=filepaths['featfile'], 
        statfile=filepaths['statfile'], 
        params=params)
    
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
    train_dataset, train_loader = get_batch_and_loader(train_patient_list, fetch_data, params, shuffle=False)
    validate_dataset, validate_loader = get_batch_and_loader(validate_patient_list, fetch_data, params, shuffle=False)
    test_dataset, test_loader = get_batch_and_loader(test_patient_list, fetch_data, params, shuffle=False)
    params['num_features_static'] = len(fetch_data.static_features)
    
    model = GNN(
        num_features_static_graph   = params['num_features_static'], 
        hidden_dim                  = params['hidden_dim'], 
        hidden_dim_2                = params['hidden_dim_2'],
        hidden_layers               = params['hidden_layers'],
        gnn_layer                   = params['gnn_layer'], 
        pooling_method              = params['pooling_method'], 
        dropout_rate                = params['dropout_rate'], 
        ratio                       = params['ratio'],
        self_loops                  = params['add_self_loops'])

    model_path = '{}/{}_model.pth'.format(params['outpath'], params['outname'])
    results_path = '{}/{}_results.csv'.format(params['outpath'], params['outname'])
    stats_path = '{}/{}_stats.csv'.format(params['outpath'], params['outname'])

    print('STARTING MODEL TRAIN/TEST')  

    if params['explainability_mode']:
        results = pd.read_csv(results_path)
        stats = pd.read_csv(stats_path)
        threshold = float(stats[stats['name']=='threshold']['value'])
        # select graphs to explain
        if params['num_positive_samples']=='all':
            exp_patient_list = test_patient_list
        else: 
            samples = explainability.sampling(results, num_positive_samples=params['num_positive_samples'], uncertainty_rate=0.8)
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
        explainability.gnn_explainer(model, exp_loader, exp_patient_list, params, threshold)
    
    elif params['tuning_mode']:
        # free up memory no longer needed
        del fetch_data 
        del train_dataset
        del validate_dataset
        del test_dataset
        
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: hyperparameter_tuning(trial, train_loader, validate_loader, params), n_trials=10)
        
        # Get the best hyperparameters
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)

        
    else:
        # normal training model
        # free up memory no longer needed
        del fetch_data 
        del train_dataset
        del validate_dataset
        del test_dataset

        # model training
        START = time.time()
        model, threshold = train_model(model, train_loader, validate_loader, params)
        torch.save(model.state_dict(), model_path)
        END = time.time()
        params['num_parameters'] = sum(p.numel() for p in model.parameters())
        params['threshold'] = threshold
        params['training_time'] = (END-START)/60

        # model testing
        results, metric_results = test_model(model, test_loader, threshold, params)
        results['target_id'] = test_patient_list
        results.to_csv(results_path, index=None)
        params.update(metric_results)
        stats = pd.DataFrame({'name':list(params.keys()), 'value':list(params.values())})
        stats.to_csv(stats_path, index=None)

