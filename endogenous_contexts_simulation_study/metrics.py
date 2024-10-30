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
    """
    Function to retrieve results from a saved file based on the parameter setup and repeat index.
    
    Parameters:
    - para_setup: Tuple containing the setup parameters.
    - repeat: Integer indicating the repeat index to load results from.
    
    Returns:
    - results: The results loaded from the pickle file based on the setup and repeat.
    """
    name_string = generate_name_from_params(para_setup)
    file_name = para_setup[-1] + '/' + name_string + '/%s_%s' % (name_string, repeat) + '.dat'
    results = pickle.load(open(file_name, 'rb'), encoding='latin1')
    return results


def get_metrics(para_setup, metrics_result_path):
    """
    Function to get aggregated metrics for a given parameter setup.
    
    Parameters:
    - para_setup: Tuple containing the setup parameters.
    - metrics_result_path: String indicating the path where the metrics results are to be saved.
    
    Returns:
    - metrics_dict_regimes, metrics_dict_union, avg_metrics_dict: Dictionaries containing the metrics for regime-specific, union graphs, and average across regimes.
    """
    N, density, max_lag, pc_alpha, sample_size, regime_children_known, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, use_regressionci, save_folder = para_setup

    results = []

    if regime_children_known == True:
        regime_known = 'True'
    elif regime_children_known == False:
        regime_known = None
    elif regime_children_known == 'and_parents':
        regime_known = 'and_parents'
        
    failed_res = 0
    for repeat in range(nb_repeats):
        try:
            res = get_results(para_setup, repeat)
        except:
            res = None
        if res is not None:
            results.append(res)
        else:
            failed_res += 1
            
    file_path = os.path.join(metrics_result_path, 'failed.txt')
    
    with open(file_path, 'a') as file:
        file.write(str(para_setup) + ' failed ' + str(failed_res) + '\n')
        
    if len(results) > 0:
        metrics_dict_regimes, intersection_regime_graphs, avg_metrics_dict = calculate_metrics_regimes(results, N=N, regime_known=regime_known, max_lag=max_lag)
        metrics_dict_union = calculate_metrics_union(results, intersection_regime_graphs, N=N, regime_known=regime_known, max_lag=max_lag)

        return metrics_dict_regimes, metrics_dict_union, avg_metrics_dict
    else:
        print('NO RESULTS FOR', para_setup)
        return None, None, None


def _get_match_score(true_link, pred_link):
    """
    Helper function to match true and predicted links in the graph.
    
    Parameters:
    - true_link: String representing the true link in the graph.
    - pred_link: String representing the predicted link in the graph.
    
    Returns:
    - count: Integer score indicating the number of matching edge marks between true and predicted links.
    """
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
    """
    Generate masks based on regime conditions and parent nodes which are used to 
    ignore the links to and from the regime indicator in the computation of the 
    metrics.
    
    Parameters:
    - n_nodes: Integer, number of nodes in the graph.
    - regime_indicator: Tuple indicating the regime variables.
    - max_lag: Integer, the maximum lag in the graph (default is 0).
    
    Returns:
    - regime_mask: Numpy array of shape (n_nodes, n_nodes, max_lag) indicating the regime mask.
    """
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
    """
    Generate masks for different regimes which are used to ignore the links from the 
    regime to its children.
    
    Parameters:
    - n_nodes: Integer, number of nodes in the graph.
    - regime_indicators: List of tuples indicating the regime variables for each regime.
    - true_graphs: Numpy array representing the true graphs for comparison.
    - max_lag: Integer, the maximum lag in the graph (default is 0).
    
    Returns:
    - regime_masks: Numpy array of shape (number of graphs, n_nodes, n_nodes, max_lag) indicating the regime masks.
    """
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
    """
    Compute various graph comparison metrics between true and predicted graphs.
    
    Parameters:
    - true_graphs: Numpy array representing the true graphs.
    - pred_graphs: Numpy array representing the predicted graphs.
    - masks: Optional; Numpy array of masks to apply to the graphs before comparison.
    
    Returns:
    - Various metrics including false positive rate (fpr), true positive rate (tpr),
      adjacency precision (adj_prec), adjacency recall (adj_rec), edge precision (edge_prec),
      edge recall (edge_rec), and F1 score (f1).
    """
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

def calculate_metrics_regimes(results, regime_known=None, boot_samples=200, N=2, max_lag=0):
    """
    Calculate regime-specific metrics based on results from multiple runs.
    
    Parameters:
    - results: List of dictionaries containing the results for each run.
    - regime_known: Optional; specifies if the regime is known, i.e., whether 
        all links should be evaluated for the metrics (regime_known = False), 
        whether the links from regime to its children should be ignored (regime_known = True)
        or whether the links to and from the regime indicator should be ignored (regime_known = 'and_parents')
    - boot_samples: Integer, the number of bootstrap samples to use (default is 200).
    - N: Integer, the number of nodes in the graph.
    - max_lag: Integer, the maximum lag in the graph (default is 0).
    
    Returns:
    - metrics_list: List of dictionaries containing the metrics for each regime.
    - intersection_regime_graphs: List of graphs representing the intersection of regime-specific predictions.
    - avg_metrics_dict: Dictionary containing the average metrics across all regimes.
    """
    regime_indicators = [res['regime_indicator'] for res in results]
    
    true_regime_graphs = {}
    true_regime_links = [res['true_regime_links_with_regime_ind'] for res in results]
    nb_regimes = len(true_regime_links[0])
    
    for regime in range(nb_regimes):
        true_g = [toys.links_to_graph(links[regime], tau_max=max_lag) for links in true_regime_links]
        true_regime_graphs[regime] = np.stack(true_g)

    true_union_graphs = [unionize_graphs([true_regime_graphs[regime][i] for regime in range(nb_regimes)], nb_regimes) for i in range(len(true_regime_graphs[0]))]
    intersection_regime_graphs = []
    metrics_list = []
    
    pool_pred_graphs = np.stack([res['union_graph_pcmci'] for res in results if res['union_graph_pcmci'] is not None])
    # true_union_graphs = np.stack([true_union_graphs[i] for i, res in enumerate(results) if res['union_graph' + key] is not None])

    
    for regime in range(nb_regimes):
        metrics_dict = dict()
        for key in ['_regimes', '_ymask']:
            
            # metrics for regime-specific graphs
            true_graphs = np.stack([graph for i, graph in enumerate(true_regime_graphs[regime]) if len(results[i]['graphs' + key])==nb_regimes])
            pred_graphs = np.stack([res['graphs' + key][regime] for res in results if len(res['graphs' + key])==nb_regimes])
            
            
            computation_time = np.array([res['computation_time' + key] for res in results if len(res['graphs' + key])==nb_regimes])
            n_realizations = computation_time.shape[0]
            if regime_known == 'and_parents':
                masks = np.stack([get_regime_and_parents_masks(N, regime_indicators[i][0], max_lag=max_lag) for i, res in enumerate(results) if len(res['graphs' + key])==nb_regimes])
            elif regime_known == 'True':
                masks = get_regime_masks(N, regime_indicators, true_graphs, max_lag=max_lag) 
            else:
                masks = None

            fpr, tpr, adj_prec, adj_rec, edge_prec, edge_rec, f1 = compute_measures(true_graphs, pred_graphs, masks)
            
            computation_time = (np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))

            metrics_dict['fpr' + key] = fpr
            metrics_dict['tpr' + key] = tpr
            metrics_dict['adj_prec' + key] = adj_prec
            metrics_dict['adj_rec' + key] = adj_rec
            metrics_dict['edge_prec' + key] = edge_prec
            metrics_dict['edge_rec' + key] = edge_rec
            metrics_dict['f1' + key] = f1
            metrics_dict['computation_time' + key] = computation_time
           
            if key == '_regimes':
                # intersection_pred = np.asarray([intersect_graphs([pred_graphs[i], pool_pred_graphs[i]], regime_indicators[i]) for i in range(len(pred_graphs))])
                intersection_pred = np.stack([intersect_graphs([pred_graphs[i], pool_pred_graphs[i]], regime_indicators[i]) for i in range(len(pred_graphs))])
                
                intersection_regime_graphs.append(intersection_pred)
                # print([intersection_pred[:, regime_indicators[i], :, :] for i in range(len(intersection_pred))])
                fpr_int, tpr_int, adj_prec_int, adj_rec_int, edge_prec_int, edge_rec_int, f1_int = compute_measures(true_graphs, intersection_pred, masks)

                metrics_dict['fpr_intersection'] = fpr_int
                metrics_dict['tpr_intersection'] = tpr_int
                metrics_dict['adj_prec_intersection'] = adj_prec_int
                metrics_dict['adj_rec_intersection'] = adj_rec_int
                metrics_dict['edge_prec_intersection'] = edge_prec_int
                metrics_dict['edge_rec_intersection'] = edge_rec_int
                metrics_dict['f1_intersection'] = f1_int
                metrics_dict['computation_time_intersection'] = computation_time
        
        metrics_list.append(metrics_dict)

    # Compute averages for the regimes
    avg_metrics_dict = dict()
    for metric in metrics_list[0].keys():
        if 'computation_time' not in metric:
            reg_numerator, reg_denominator = np.asarray([metrics_list[regime][metric][0] for regime in range(nb_regimes)]), np.asarray([metrics_list[regime][metric][1] for regime in range(nb_regimes)])
            n_realizations = reg_numerator.shape[1]
            avg_reg_numerator, avg_reg_denominator = np.mean(reg_numerator, axis=0), np.mean(reg_denominator, axis=0)
            
            metric_boot = np.zeros(boot_samples)
            for b in range(boot_samples):
                # Store the unsampled values in b=0
                rand = np.random.randint(0, n_realizations, n_realizations)
                metric_boot[b] = avg_reg_numerator[rand].sum() / avg_reg_denominator[rand].sum()
            if avg_reg_denominator.sum() > 0:
                avg_metrics_dict[metric] = (avg_reg_numerator.sum() / avg_reg_denominator.sum(), metric_boot.std())
            else: 
                avg_metrics_dict[metric] = (0., metric_boot.std())
        elif 'computation_time' in metric:
            avg_val = np.asarray([metrics_list[regime][metric] for regime in range(nb_regimes)])
            avg_metrics_dict[metric] = avg_val
    
    for metrics_dict in metrics_list:     
       for metric in metrics_dict.keys():
            if 'computation_time' not in metric:
                # if 'computation_time' not in metric:
                numerator, denominator = metrics_dict[metric]
                n_realizations = numerator.shape[0]
                metric_boot = np.zeros(boot_samples)
                for b in range(boot_samples):
                    # Store the unsampled values in b=0
                    rand = np.random.randint(0, n_realizations, n_realizations)
                    metric_boot[b] = numerator[rand].sum() / denominator[rand].sum()
                if denominator.sum() > 0:
                    metrics_dict[metric] = (numerator.sum() / denominator.sum(), metric_boot.std())
                else: 
                    metrics_dict[metric] = (0., metric_boot.std())

    return metrics_list, intersection_regime_graphs, avg_metrics_dict


def calculate_metrics_union(results, intersection_regime_graphs=None, boot_samples=200, N=2, regime_known=None, max_lag=0):
    """
    Calculate metrics for the union of graphs from different regimes.
    
    Parameters:
    - results: List of dictionaries containing the results for each run.
    - intersection_regime_graphs: Optional; List of graphs for the PC-B method.
    - boot_samples: Integer, the number of bootstrap samples to use (default is 200).
    - N: Integer, the number of nodes in the graph.
    - regime_known: Optional; specifies if the regime is known, i.e., whether 
        all links should be evaluated for the metrics (regime_known = False), 
        whether the links from regime to its children should be ignored (regime_known = True)
        or whether the links to and from the regime indicator should be ignored (regime_known = 'and_parents')
    - max_lag: Integer, the maximum lag in the graph (default is 0).
    
    Returns:
    - metrics_dict: Dictionary containing the metrics for the union of graphs across all regimes.
    """
    true_regime_links = [res['true_regime_links_with_regime_ind'] for res in results]

    regime_indicators = [res['regime_indicator'] for res in results]
    
    nb_regimes = len(true_regime_links[0])
    true_regime_graphs = {}
    for regime in range(nb_regimes):
        true_g = [toys.links_to_graph(links[regime], tau_max=max_lag) for links in true_regime_links]
        true_regime_graphs[regime] = np.stack(true_g)
    
    true_union_graphs = np.stack([unionize_graphs([true_regime_graphs[regime][i] for regime in range(nb_regimes)], nb_regimes) for i in range(len(true_regime_graphs[0]))])
    
    pred_pcmci_graphs = np.stack([res['union_graph_pcmci'] for res in results])
    
    N, N, max_lag = true_union_graphs[0].shape

    metrics_dict = dict()
    for key in ['_pcmci', '_regimes', '_ymask']:
        pred_graphs = np.stack([res['union_graph' + key] for res in results if res['union_graph' + key] is not None])
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

    
    if regime_known == 'and_parents':
        masks = np.stack([get_regime_and_parents_masks(N, regime_indicators[i][0], max_lag=max_lag) for i, res in enumerate(results)])
    elif regime_known == True:
        masks = get_regime_masks(N, regime_indicators, true_union_graphs, max_lag=max_lag) 
    else:
        masks = None
        
    intersection_union_graphs = np.stack([unionize_graphs([intersection_regime_graphs[regime][i] for regime in range(nb_regimes)], nb_regimes) for i in range(len(intersection_regime_graphs[0]))])
    fpr_int, tpr_int, adj_prec_int, adj_rec_int, edge_prec_int, edge_rec_int, f1_int = compute_measures(true_union_graphs, intersection_union_graphs, masks)
    
    metrics_dict['union_fpr_intersection'] = fpr_int
    metrics_dict['union_tpr_intersection'] = tpr_int
    metrics_dict['union_adj_prec_intersection'] = adj_prec_int
    metrics_dict['union_adj_rec_intersection'] = adj_rec_int
    metrics_dict['union_edge_prec_intersection'] = edge_prec_int
    metrics_dict['union_edge_rec_intersection'] = edge_rec_int    
    metrics_dict['union_f1_intersection'] = f1_int    
    

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

    # config_path = './../configs/test_config.yaml'  # Path to your YAML configuration file
    config_path = args.yaml_path
    results_folder, all_configurations = generate_configurations(config_path)

    for configuration in all_configurations:
          
        result_path = configuration[0][-1] + '/' + configuration[1]
        # metrics_result_path = configuration[0][-1] + '/metrics_v3/'
        
        if args.save_results_folder is not None:
            metrics_result_path = args.save_results_folder
        else:
            metrics_result_path = configuration[0][-1] + '/metrics_v3/'

        if not os.path.exists(metrics_result_path):
            os.makedirs(metrics_result_path)
        

        file_name_union = metrics_result_path + generate_name_from_params(configuration[0]) + '_union.dat'
        file_name_regimes = metrics_result_path + generate_name_from_params(configuration[0]) + '_regimes.dat'
        file_name_avg = metrics_result_path + generate_name_from_params(configuration[0]) + '_avg_regimes.dat'
        
        

        metrics_dict_regimes, metrics_dict_union, avg_metrics_dict = get_metrics(configuration[0], metrics_result_path)
        
        if metrics_dict_union is not None:
            # for metric in metrics_dict_union:
            file = open(file_name_union, 'wb')
            pickle.dump(metrics_dict_union, file, protocol=-1)
            file.close()
        if metrics_dict_regimes is not None:
            # for metric in metrics_dict_regimes:
            file = open(file_name_regimes, 'wb')
            pickle.dump(metrics_dict_regimes, file, protocol=-1)
            file.close()    
        if avg_metrics_dict is not None:
            # for metric in avg_metrics_dict:
            file = open(file_name_avg, 'wb')
            pickle.dump(avg_metrics_dict, file, protocol=-1)
            file.close()    