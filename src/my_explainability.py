import pandas as pd
from tqdm import tqdm
import numpy as np
import torch


def sampling(results, num_positive_samples, uncertainty_rate=0.8):

    results['index'] = range(len(results))
    # patients the model predicted correctly and with low uncertainty
    results = results[results['actual']==results['pred_binary']]
    results = results.sort_values(by='pred_raw_se')[0:int(len(results)*uncertainty_rate)]

    # randomly sample at a rate of 50% minority class, 50% majority class
    if len(results[results['pred_binary']==1])<num_positive_samples:
        print("Only {}<{} positive samples were identified".format(len(results[results['pred_binary']==1]), num_positive_samples))
        num_positive_samples = len(results[results['pred_binary']==1])

    results = results.sample(frac=1)
    positive_samples = results[results['pred_binary']==1][0:num_positive_samples]['index'].tolist()
    negative_samples = results[results['pred_binary']==0][0:num_positive_samples]['index'].tolist()
    samples = positive_samples + negative_samples

    print("Returning {} positive samples and {} negative samples".format(len(positive_samples), len(negative_samples)))
    return samples



def gnn_explainer(model, exp_loader, patient_list, params, threshold):

    outfile = '{}/{}_explainer'.format(params['outpath'], params['outname'])
    
    with open(f'{outfile}_features.csv', 'w') as f_features:
        f_features.write("target_id,feature_masked,new_score,old_score,delta\n")
        with open(f'{outfile}_nodes.csv', 'w') as f_nodes:
            f_nodes.write("target_id,node_masked,new_score,old_score,delta\n")

            counter = 0
            mask_value = -1
            # assumes one graph is loaded at a time
                    
            for data_batch in tqdm(exp_loader, total=len(patient_list)):
              
                n_nodes                 = data_batch.x.shape[0]
                n_features              = data_batch.x.shape[1]
                x                       = data_batch.x.to(params['device'])
                edge_index              = data_batch.edge_index.to(params['device'])
                edge_weight             = data_batch.edge_attr.to(params['device'])
                batch                   = data_batch.batch.to(params['device'])
                target_index            = data_batch.target_index.to(params['device'])
                
                # get baseline GNN score
                target_id = patient_list[counter].item()
                with torch.no_grad():
                    if params['use_edge']=='True':
                        baseline_score = model(x, edge_index, edge_weight, batch, target_index).item()
                    else:
                        baseline_score = model(x, edge_index, None, batch, target_index).item()
                
                # remove one feature at the time and evaluate the change in the GNN mdoel score
                for feature in np.arange(n_features):
                    x_masked_feature = x.clone()
                    x_masked_feature[:, feature] = mask_value
                    x_masked_feature.to(params['device'])
                    with torch.no_grad():
                        # look in forward() in model.py for model architecture
                        if params['use_edge']=='True':
                            output = model(x, edge_index, edge_weight, batch, target_index).item()
                        else:
                            output = model(x, edge_index, None, batch, target_index).item()
                    
                    #compare result to baseline score
                    delta = output - baseline_score
                    results = [str(target_id), str(int(feature)), str(round(output,4)), str(round(baseline_score,4)), str(round(delta,4))]
                    f_features.write(','.join(results)+'\n')
                    
                # remove one node at the time and evaluate the change in the GNN mdoel score
                for node in np.arange(n_nodes):
                    x_masked_node = x.clone()
                    x_masked_node[node, :] = mask_value
                    x_masked_node.to(params['device'])
                    with torch.no_grad():
                        # look in forward() in model.py for model architecture
                        if params['use_edge']=='True':
                            output = model(x_masked_node, edge_index, edge_weight, batch, target_index).item()
                        else:
                            output = model(x_masked_node, edge_index, None, batch, target_index).item()
                    #compare result to baseline score
                    delta = output - baseline_score
                    results = [str(target_id), str(int(node)), str(round(output,4)), str(round(baseline_score,4)), str(round(delta,4))]
                    f_nodes.write(','.join(results)+'\n')
                            
                counter += 1
