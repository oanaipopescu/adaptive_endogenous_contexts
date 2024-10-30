import copy

import numpy as np
import pickle
import os
from tigramite.toymodels import structural_causal_processes as toys
from utils import links_to_cyclic_graph
from generate_data import unionize_graphs, intersect_graphs
from matplotlib import pyplot as plt

import argparse

from config_generator import generate_configurations, generate_string_from_params, generate_name_from_params

import tigramite.plotting as tp

def get_results(para_setup, repeat):
    name_string = generate_name_from_params(para_setup)
    file_name = para_setup[-1] + '/' + name_string + '/%s_%s' % (name_string, repeat) + '_fci.dat'
    # print('load  ', file_name)
    results = pickle.load(open(file_name, 'rb'), encoding='latin1')
    return results


def get_metrics(para_setup):
    N, density, max_lag, pc_alpha, sample_size, regime_children_known, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, use_regressionci, save_folder = para_setup

    results = []

    if regime_children_known == True:
        regime_known = 'True'
    elif regime_children_known == False:
        regime_known = None
    elif regime_children_known == 'and_parents':
        regime_known = 'and_parents'
    
    for repeat in range(nb_repeats):
        try:
            res = get_results(para_setup, repeat)
        except:
            res = None
        if res is not None:
            results.append(res)
    
    if len(results) > 0:
        metrics_dict_union = calculate_metrics_union(results, N=N, regime_known=regime_known, max_lag=max_lag)

        return metrics_dict_union
    else:
        print('NO RESULTS FOR', para_setup)
        return None


def _get_match_score(true_link, pred_link):
    if true_link == "" or pred_link == "": return 0
    count = 0
    # If left edgemark is correct add 1
    if true_link[0] == pred_link[0]:
        count += 1
    # If right edgemark is correct add 1
    if true_link[2] == pred_link[2]:
        count += 1
    return count


match_func = np.vectorize(_get_match_score, otypes=[int])


def get_regime_and_parents_masks(n_nodes, regime_indicator, max_lag=0):
    if max_lag == 0:
        max_lag = 1  # Ensure there is at least one lag dimension

    # Initialize the regime_mask with all True values
    regime_mask = np.ones((n_nodes, n_nodes, max_lag)).astype('bool')
    
    # Set the lower triangular part of the first lag slice to True (others are already True)
    # Making it symmetrical by applying lower triangular to both lower and upper (transposed) parts
    tril_mask = np.tril(np.ones((n_nodes, n_nodes)), k=-1).astype('bool')
    regime_mask[:, :, 0] = tril_mask | tril_mask.T

    # Apply regime conditions symmetrically
    # Disabling both rows and columns for the nodes in regime_indicator
    regime_mask[regime_indicator, :, :] = False
    regime_mask[:, regime_indicator, :] = False
    
    return regime_mask

def get_regime_masks(n_nodes, regime_indicators, true_graphs, max_lag=0):
    if max_lag == 0:
        max_lag = 1  # Ensure there is at least one lag dimension

    regime_masks = np.zeros((true_graphs.shape[0], n_nodes, n_nodes, 1)).astype(bool)

    for i in range(len(true_graphs)):
        regime_mask = np.ones((n_nodes, n_nodes, max_lag)).astype('bool')
        tril_mask = np.tril(np.ones((n_nodes, n_nodes)), k=-1).astype('bool')
        regime_mask[:, :, 0] = tril_mask | tril_mask.T
        for k in range(n_nodes):
            for j in range(n_nodes):
                if k != j and k == regime_indicators[i][0]: # set only outgoing
                    if true_graphs[i, k, j, :] == '-->':
                        regime_mask[k, j] = regime_mask[j, k] = False
        regime_masks[i, :, :, :] = regime_mask
        
    return regime_masks

def compute_measures(true_graphs, pred_graphs, masks=None):
    if masks is not None:
        for i in range(len(pred_graphs)):
            pred_graphs[i] = np.where(masks[i], pred_graphs[i], "")
            true_graphs[i] = np.where(masks[i], true_graphs[i], "")
    
    fpr = (((true_graphs == "") * (pred_graphs != "")).sum(axis=(1, 2, 3)), (true_graphs == "").sum(axis=(1, 2, 3)))
    tpr = (((true_graphs != "") * (pred_graphs != "")).sum(axis=(1, 2, 3)), (true_graphs != "").sum(axis=(1, 2, 3)))
    adj_prec =  ((((true_graphs != "") * (pred_graphs != "")).sum(axis=(1,2,3)), ((pred_graphs != "")).sum(axis=(1,2,3))))
    adj_rec = (((true_graphs != "") * (pred_graphs != "")).sum(axis=(1,2,3)), ((true_graphs != "")).sum(axis=(1,2,3)))

    edge_prec = ((match_func(true_graphs, pred_graphs)).sum(axis=(1, 2, 3)), (2. * (pred_graphs != "")).sum(axis=(1,2,3)))
    edge_rec = ((match_func(true_graphs, pred_graphs)).sum(axis=(1, 2, 3)),  (2. * (true_graphs != "")).sum(axis=(1, 2, 3)))
    
    f1 = (2 * ((adj_prec[0] / adj_prec[1]) * (adj_rec[0] / adj_rec[1])), (adj_prec[0] /adj_prec[1])  + (adj_rec[0] / adj_rec[1]))

    return fpr, tpr, adj_prec, adj_rec, edge_prec, edge_rec, f1


def calculate_metrics_union(results, boot_samples=200, N=2, regime_known=None, max_lag=0):
    true_regime_links = [res['true_regime_links_with_regime_ind'] for res in results]

    regime_indicators = [res['regime_indicator'] for res in results]
    
    nb_regimes = len(true_regime_links[0])
    true_regime_graphs = {}
    for regime in range(nb_regimes):
        true_g = [toys.links_to_graph(links[regime], tau_max=max_lag) for links in true_regime_links]
        true_regime_graphs[regime] = np.stack(true_g)
    
    true_union_graphs = np.stack([unionize_graphs([true_regime_graphs[regime][i] for regime in range(nb_regimes)], nb_regimes) for i in range(len(true_regime_graphs[0]))])
    
    N, N, max_lag = true_union_graphs[0].shape

    metrics_dict = dict()
    for key in ['_fci', '_nod']:

        pred_graphs = np.stack([toys.links_to_graph(res['union_graph' + key]) for res in results if res['union_graph' + key] is not None])
        true_graphs = np.stack([true_union_graphs[i] for i, res in enumerate(results) if res['union_graph' + key] is not None])

        computation_time = np.array([res['computation_time' + key] for res in results if res['union_graph' + key] is not None])                                   

        n_realizations = computation_time.shape[0] # we left some of the realizations out

        if regime_known == 'and_parents':
            masks = np.stack([get_regime_and_parents_masks(N, regime_indicators[i][0], max_lag=max_lag) for i, res in enumerate(results)])
        elif regime_known == 'True':
            masks = get_regime_masks(N, regime_indicators, true_graphs, max_lag=max_lag) 
        else:
            masks = None
        fpr, tpr, adj_prec, adj_rec, edge_prec, edge_rec, f1 = compute_measures(true_graphs, pred_graphs, masks)
        
        comp_time = (np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))

        metrics_dict['union_fpr' + key] = fpr
        metrics_dict['union_tpr' + key] = tpr
        metrics_dict['union_adj_prec' + key] = adj_prec
        metrics_dict['union_adj_rec' + key] = adj_rec
        metrics_dict['union_edge_prec' + key] = edge_prec
        metrics_dict['union_edge_rec' + key] = edge_rec
        metrics_dict['union_f1' + key] = f1
        
        metrics_dict['computation_time' + key] = comp_time

    for metric in metrics_dict.keys():
        if 'computation_time' not in metric:
            numerator, denominator = metrics_dict[metric]
            n_realizations = numerator.shape[0]
            metric_boot = np.zeros(boot_samples)
            for b in range(boot_samples):
                # Store the unsampled values in b=0
                rand = np.random.randint(0, n_realizations, n_realizations)
                metric_boot[b] = numerator[rand].sum() / denominator[rand].sum()

            metrics_dict[metric] = (numerator.sum() / denominator.sum(), metric_boot.std())

    return metrics_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate configurations from a YAML file.")
    parser.add_argument('yaml_path', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('save_results_folder', type=str, help='Folder to save the metrics.')
    args = parser.parse_args()

    config_path = args.yaml_path
    results_folder, all_configurations = generate_configurations(config_path)

    for configuration in all_configurations:
          
        result_path = configuration[0][-1] + '/' + configuration[1]
        
        if args.save_results_folder is not None:
            metrics_result_path = args.save_results_folder
        else:
            metrics_result_path = configuration[0][-1] + '/metrics_v3/'

        if not os.path.exists(metrics_result_path):
            os.makedirs(metrics_result_path)
        

        file_name_union = metrics_result_path + generate_name_from_params(configuration[0]) + '_union_fci.dat'
        

        metrics_dict_union = get_metrics(configuration[0])
        
        if metrics_dict_union is not None:
            # for metric in metrics_dict_union:
            file = open(file_name_union, 'wb')
            pickle.dump(metrics_dict_union, file, protocol=-1)
            file.close()