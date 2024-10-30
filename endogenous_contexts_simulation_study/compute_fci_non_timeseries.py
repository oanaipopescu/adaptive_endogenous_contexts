import numpy as np
import time
import sys
import socket
import os
import mpi
import copy
import pickle
import time
from itertools import permutations, combinations
from typing import Dict, List, Optional, Set, Tuple, Dict

import networkx as nx

from numpy import ndarray

import argparse 

from matplotlib import pyplot as plt
from numpy.random import SeedSequence
import networkx as nx

from endo_regime_pcmci import EndoRegimePCMCI, MixedTestPCMCI
import generate_data as gd
from config_generator import generate_configurations, generate_string_from_params, generate_name_from_params

from tigramite.pcmci import PCMCI
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import data_processing as pp
import tigramite.plotting as tp

# import indepenence tests
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from cmiknnmixed import CMIknnMixed
from tigramite.independence_tests.regressionCI import RegressionCI

import warnings
from queue import Queue
from numpy import ndarray

import pydot 

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from causallearn.utils.FAS import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import SkeletonDiscovery, UCSepset, Meek
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn.search.ConstraintBased.PC import get_parent_missingness_pairs, skeleton_correction
from causallearn.utils.cit import *
from causallearn.search.ConstraintBased.FCI import *



########## SET THESE BEFORE RUNNING!
sequential = False
plot_data = True

try:
    arg = sys.argv
    num_cpus = int(arg[1])
    samples = int(arg[2])
    config_list = list(arg)[3:]
    num_configs = len(config_list)
except Exception as e:
    print(e)
    arg = ''
    num_cpus = 2
    samples = 100
    verbosity = 2
    config_list = []
    # print config_list


time_start = time.time()


def unpack_params(params_str):
    if sequential:
        para_setup_string = params_str
    else:
        para_setup_string, sam = params_str

    paras = para_setup_string.split('-')
    # paras = [w.replace("'", "") for w in paras]

    N = int(paras[0])
    density = float(paras[1])
    max_lag = int(paras[2])
    pc_alpha = float(paras[3])
    sample_size = int(paras[4])
    regime_children_known = str(paras[5])
    nb_changed_links = int(paras[6])
    nb_regimes = int(paras[7])
    nb_repeats = int(paras[8])
    cycles_only = paras[9].lower() in ['true']
    remove_only = paras[10].lower() in ['true']
    use_cmiknnmixed = paras[11].lower() in ['true']
    imbalance_factor =  float(paras[12])
    save_folder = str(paras[13])
    
    return N, density, max_lag, pc_alpha, sample_size, regime_children_known, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, imbalance_factor, save_folder


def add_children_links(links, regime_children, regime_indicator):
    for regime in range(len(links)):
        for var, lag in regime_children:
            links[regime][var].append((regime_indicator, 1.0, 'regime'))

    return links

def fci_custom_CIT(dataset, independence_test_method, link_assumptions, alpha: float = 0.05, depth: int = -1,
        max_path_length: int = -1, verbose: bool = False, background_knowledge: BackgroundKnowledge | None = None, show_progress: bool = False,
        **kwargs) -> Tuple[Graph, List[Edge]]:
    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    # independence_test_method = CIT(dataset, method=independence_test_method, **kwargs)

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(background_knowledge) != BackgroundKnowledge:
        raise TypeError("'background_knowledge' must be 'BackgroundKnowledge' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------


    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i}")
        node.add_attribute("id", i)
        nodes.append(node)

    if link_assumptions is not None:
        background_knowledge = BackgroundKnowledge()
        for i in range(dataset.shape[1]):
            for j in range(dataset.shape[1]):
                if link_assumptions[i][(j, 0)] == '-->':
                    background_knowledge.add_required_by_node(nodes[i], nodes[j])
    else:
        background_knowledge = None

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    graph, sep_sets, test_results = fas(dataset, nodes, independence_test_method=independence_test_method, alpha=alpha,
                                        knowledge=background_knowledge, depth=depth, verbose=verbose, show_progress=False)

    reorientAllWith(graph, Endpoint.CIRCLE)

    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    removeByPossibleDsep(graph, independence_test_method, alpha, sep_sets)

    reorientAllWith(graph, Endpoint.CIRCLE)
    rule0(graph, nodes, sep_sets, background_knowledge, verbose)

    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
        change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

        if change_flag or (first_time and background_knowledge is not None and
                           len(background_knowledge.forbidden_rules_specs) > 0 and
                           len(background_knowledge.required_rules_specs) > 0 and
                           len(background_knowledge.tier_map.keys()) > 0):
            change_flag = ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  background_knowledge, verbose)

            first_time = False

            if verbose:
                print("Epoch")

    graph.set_pag(True)

    edges = get_color_edges(graph)

    return graph, edges


class CustomCIT(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        # self.check_cache_method_consistent('fisherz', NO_SPECIFIED_PARAMETERS_MSG)
        self.method = 'regCI'
        self.parcorr = RobustParCorr()
        self.regCI = RegressionCI()
        
        self.assert_input_data_is_valid()

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        
        x_array = copy.deepcopy(self.data[:, Xs])
        y_array = copy.deepcopy(self.data[:, Ys])
        z_array = copy.deepcopy(self.data[:, condition_set])

        # print('unique', len(np.unique(x_array[:, 0])))
        x_type = np.zeros(x_array.shape)
        y_type = np.zeros(y_array.shape)
        z_type = np.zeros(z_array.shape)

        for i in range(x_array.shape[1]):
            if len(np.unique(x_array[:, i])) < 5:
                x_type[:, i] = 1
                
        for i in range(y_array.shape[1]):
            if len(np.unique(y_array[:, i])) < 5:
                y_type[:, i] = 1

        for i in range(z_array.shape[1]):
            if len(np.unique(z_array[:, i])) < 5:
                z_type[:, i] = 1

        array = np.concatenate([x_array, y_array, z_array], axis=-1)
        xyz = np.concatenate([[0] * x_array.shape[1], [1] * y_array.shape[1], [2] * z_array.shape[1]])
        data_type = np.concatenate([x_type, y_type, z_type], axis=-1)

        T, dim = array.shape[0], array.shape[1]
        if np.sum(data_type) == 0:
            meas = self.parcorr.get_dependence_measure(array.T, xyz, data_type=data_type.T)
            p = self.parcorr.get_analytic_significance(meas, T, dim, xyz)
        else:
            meas = self.regCI.get_dependence_measure(array.T, xyz, data_type=data_type.T)
            p = self.regCI.get_analytic_significance(meas, T, dim, xyz)

        return p
        
def custom_cdnod(data, indep_test, alpha: float=0.05, stable: bool=True,
          uc_rule: int=0, uc_priority: int=2, mvcdnod: bool=False, correction_name: str='MV_Crtn_Fisher_Z',
          background_knowledge: Optional[BackgroundKnowledge]=None, verbose: bool=False,
          show_progress: bool = False, **kwargs) -> CausalGraph:
    # augment the variable set by involving c_indx to capture the distribution shift
    
    return custom_cdnod_alg(data, indep_test, alpha=alpha, stable=stable, uc_rule=uc_rule,
                         uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                         show_progress=show_progress, **kwargs)


def custom_cdnod_alg(data, indep_test, alpha: float, stable: bool, uc_rule: int, uc_priority: int,
              background_knowledge: Optional[BackgroundKnowledge] = None, verbose: bool = False,
              show_progress: bool = False, **kwargs) -> CausalGraph:

    start = time.time()
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable, show_progress=show_progress)

    # orient the direction from c_indx to X, if there is an edge between c_indx and X
    c_indx_id = data.shape[1] - 1
    for i in cg_1.G.get_adjacent_nodes(cg_1.G.nodes[c_indx_id]):
        cg_1.G.add_directed_edge(cg_1.G.nodes[c_indx_id], i)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg


def calculate_graphs(params_str, seedSeq, plot_data=False, sam=None):
    # calculate regime-specific graphs for each regime
    # also calculate union graph (gt, PCMCI, unionize regime-spec graphs)
    random_state = np.random.default_rng(seedSeq)

    N, density, max_lag, pc_alpha, sample_size, regime_children_known, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, imbalance_factor, save_folder = unpack_params(params_str)

    folder_name = generate_name_from_params((N, density, max_lag, pc_alpha, sample_size, regime_children_known, nb_changed_links, nb_regimes, nb_repeats, cycles_only, remove_only, use_cmiknnmixed, imbalance_factor, save_folder))
    result_path = save_folder + '/' + folder_name

    metrics_result_path = result_path + '/metrics'
    figure_path = result_path + '/figures/'
    graphs_path = figure_path + '/g_' + str(sam) + '/'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(metrics_result_path):
        os.makedirs(metrics_result_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    
    child_seeds = seedSeq.spawn(nb_regimes + 3)

    # randomly select a variable to be regime indicator
    regime_indicator_var = random_state.integers(N)
    regime_indicator = (regime_indicator_var, 0)
    
    dep_coeffs = np.asarray([0.6, 0.5, 0.4, -0.4, -0.5, -0.6]) * 3

    links_base, true_regime_links, true_regime_children, causal_order = gd.generate_regime_model(N, density, child_seeds, nb_regimes, regime_indicator, max_lag, 
                                                                                     nb_changed_links, remove_only=remove_only, cycles_only=cycles_only,
                                                                                     dep_coeffs=dep_coeffs)
    if true_regime_links is None:
        print('nothing found')
        return None

    # add links of true regime children to regime graphs
    true_regime_links_with_regime_ind = add_children_links(copy.deepcopy(true_regime_links), true_regime_children,
                                                           regime_indicator)
    true_regime_links_with_regime_ind_str ={i: {var: [(val[0], val[1], "func") for val in parents] for var, parents in links.items()} for i, links in true_regime_links_with_regime_ind.items()}
    base_links = {var: [(val[0], val[1], "lin_f") for val in parents] for var, parents in links_base.items()}
    
    true_regime_links_str = {i: {var: [(val[0], val[1], "lin_f") for val in parents] for var, parents in links.items()} for i, links in true_regime_links.items()}
    true_union_links = gd.unionize_links(true_regime_links, true_regime_children, nb_regimes, regime_indicator)
    true_union_links_str = {var: [(val[0], val[1], "lin_f") for val in parents] for var, parents in true_union_links.items()}

    if plot_data:
        tp.plot_graph(toys.links_to_graph(links_base))
        plt.savefig(graphs_path + 'base_' + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')
        for r in range(len(true_regime_links_with_regime_ind)):
            tp.plot_graph(toys.links_to_graph(true_regime_links[r]))
            plt.savefig(graphs_path + 'graph_r_no_regind_' + str(r) + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')
            tp.plot_graph(toys.links_to_graph(true_regime_links_with_regime_ind[r]))
            plt.savefig(graphs_path + 'graph_r_' + str(r) + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')
        
        tp.plot_graph(toys.links_to_graph(true_union_links_str))
        plt.savefig(graphs_path + 'union_graph_' + str(r) + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')

    # generate data
    data, masks, data_type = gd.generate_regime_data(sample_size, true_regime_links, nb_regimes, child_seeds, regime_indicator,
                                                     max_lag, causal_order=causal_order, regime_thresholds=None, imbalance_factor=imbalance_factor)

    
    link_assumptions = {j: {(i, -tau): 'o?o' for i in range(N) for tau in range(max_lag+1)} for j in range(N)}
    link_assumptions[regime_indicator[0]] = {}
    for var in range(N):
        for lag in range(max_lag + 1):
            link_assumptions[var][(regime_indicator[0], -lag)] = 'o?o'
            link_assumptions[regime_indicator[0]][(var, -lag)] = 'o?o'
    for var, lag in true_regime_children:
        link_assumptions[var][regime_indicator] = '-->'
        link_assumptions[regime_indicator[0]][(var, 0)] = '<--'
    
    time_start_union = time.time()
    

    regCIT = CustomCIT(data)
    fci_g, edges = fci_custom_CIT(data, regCIT, link_assumptions)
    fci_pdy = GraphUtils.to_pydot(fci_g)
    
    # Parse the DOT data into a graph using pydot
    fci_graphs = pydot.graph_from_dot_data(fci_pdy.to_string())
    fci_dot_graph = fci_graphs[0]
    
    # Convert to a NetworkX graph
    fci_nx_graph = nx.nx_pydot.from_pydot(fci_dot_graph)
    
    # Convert to DiGraph if you need to ensure the graph is directed
    fci_nx_digraph = nx.DiGraph(fci_nx_graph)

    fci_nodes = fci_nx_digraph.nodes()
    fci_edges = fci_nx_digraph.edges()
    fci_union = {}
    for node in fci_nodes: 
        fci_union[int(node)] = []
        for edge in fci_edges:
            if edge[1] == node:
                fci_union[int(node)].append(((int(edge[0]), 0), 0.5, 'func'))

    time_end_union = time.time()

    if plot_data:
        tp.plot_graph(toys.links_to_graph(fci_union))
        plt.savefig(graphs_path + '_jci_fci_' + str(sam) + '_cycles_only_' + str(cycles_only) + '.png')

    computation_time_fci = time_end_union - time_start_union

    time_start_nod = time.time()

    regCIT2 = CustomCIT(data)
    nod_g = custom_cdnod(data, regCIT2)
    
    nod_pdy = GraphUtils.to_pydot(nod_g.G)
    
    # Parse the DOT data into a graph using pydot
    nod_graphs = pydot.graph_from_dot_data(nod_pdy.to_string())
    nod_dot_graph = nod_graphs[0]
    
    # Convert to a NetworkX graph
    nod_nx_graph = nx.nx_pydot.from_pydot(nod_dot_graph)
    
    # Convert to DiGraph if you need to ensure the graph is directed
    nod_nx_digraph = nx.DiGraph(nod_nx_graph)

    nod_nodes = nod_nx_digraph.nodes()
    nod_edges = nod_nx_digraph.edges()
    nod_union = {}
    for node in nod_nodes: 
        nod_union[int(node)] = []
        for edge in nod_edges:
            if edge[1] == node:
                nod_union[int(node)].append(((int(edge[0]), 0), 0.5, 'func'))

    time_end_nod = time.time()
    
    if plot_data:
        tp.plot_graph(toys.links_to_graph(nod_union))
        plt.savefig(graphs_path + 'cdnod' + str(sam) + '_cycles_only_' + str(cycles_only) + '.png')

    computation_time_nod = time_end_nod - time_start_nod
    
    return {
        'base_links': base_links,
        'causal_order': causal_order,
        'true_regime_links': true_regime_links_str,
        'true_regime_children': true_regime_children,
        'true_regime_links_with_regime_ind': true_regime_links_with_regime_ind_str,
        'true_union_links': true_union_links_str,
        'regime_indicator': regime_indicator,

        'computation_time_fci': computation_time_fci,
        'union_graph_fci': fci_union,
        'computation_time_nod': computation_time_nod,
        'union_graph_nod': nod_union,
    }


def process_chunks(job_id, chunk, seed):
    results = {}
    num_here = len(chunk)
    model_seeds = seed.spawn(num_here)
    time_start_process = time.time()
    for isam, config_sam in enumerate(chunk):
        # res = None
        # while res is None:
        results[config_sam] = calculate_graphs(config_sam, model_seeds[isam], plot_data=plot_data, sam=job_id)
        # results[config_sam] = res
        current_runtime = (time.time() - time_start_process) / 3600.
        current_runtime_hr = int(current_runtime)
        current_runtime_min = 60. * (current_runtime % 1.)
        estimated_runtime = current_runtime * num_here / (isam + 1.)
        estimated_runtime_hr = int(estimated_runtime)
        estimated_runtime_min = 60. * (estimated_runtime % 1.)
        print("job_id %d index %d/%d: %dh %.1fmin / %dh %.1fmin:  %s" % (
            job_id, isam + 1, num_here, current_runtime_hr, current_runtime_min,
            estimated_runtime_hr, estimated_runtime_min, config_sam))
    return results


def master():
    print("Starting with num_cpus = ", num_cpus, config_list)

    all_configs = dict([(conf, {'results': {},
                                "graphs": {},
                                "val_min": {},
                                "max_cardinality": {},

                                "true_graph": {},
                                "computation_time": {}, }) for conf in config_list])

    job_list = [(conf, i) for i in range(samples) for conf in config_list]
    num_tasks = len(job_list)
    num_jobs = min(num_cpus - 1, num_tasks)

    def split(a, n):
        k, m = len(a) // n, len(a) % n
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    config_chunks = split(job_list, num_jobs)

    print("num_tasks %s" % num_tasks)
    print("num_jobs %s" % num_jobs)

    ss = SeedSequence(12345)

    # Spawn off 10 child SeedSequences to pass to child processes.
    child_seeds = ss.spawn(len(config_chunks))

    ## Send
    for job_id, chunk in enumerate(config_chunks):
        print("submit %d / %d" % (job_id, len(config_chunks)))
        mpi.submit_call("process_chunks", (job_id, chunk, child_seeds[job_id]), id=job_id)

    ## Retrieve
    id_num = 0
    for job_id, chunk in enumerate(config_chunks):
        print("\nreceive %s" % job_id)
        tmp = mpi.get_result(id=job_id)
        for conf_sam in list(tmp.keys()):
            config = conf_sam[0]
            sample = conf_sam[1]
            # TODO: save as you get results
            para_setup_str = tuple(config.split("-"))
            config_string = generate_name_from_params(para_setup_str)
            result_path = para_setup_str[-1] + '/' + config_string
            file_name = result_path + '/%s_%s' % (config_string, id_num)
            file_name_cleaned = file_name.replace("'", "").replace('"', '') + '_fci.dat'
            print('writing... ', file_name_cleaned)
            file = open(file_name_cleaned, 'wb')
            pickle.dump(tmp[conf_sam], file, protocol=-1)
            file.close()
            id_num += 1

    time_end = time.time()
    print('Run time in hours ', (time_end - time_start) / 3600.)


mpi.run(verbose=False)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Run tasks from a YAML file.")
#     parser.add_argument('yaml_path', type=str, help='Path to the YAML configuration file.')
#     args = parser.parse_args()

#     # config_path = './../configs/test_config.yaml'  # Path to your YAML configuration file
#     config_path = args.yaml_path

#     results_folder, all_configurations = generate_configurations(config_path)
#     nb_repeats = all_configurations[0][0][8]

#     if not os.path.exists(results_folder):
#                 os.makedirs(results_folder)

#     already_there = []
#     configurations = []
    
#     for configuration in all_configurations:
#         config_params = configuration[0]
#         suffix = configuration[1]
#         save_folder = config_params[-1] + '/' + suffix

#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
        
#         current_results_files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]
        
#         if suffix not in configurations:
#             configurations.append(configuration)

#     num_configs = len(configurations)  # min(num_jobs, num_configs)  # num_configs/num_jobs

#     print("number of todo configs ", num_configs)
#     print("number of existing configs ", len(already_there))
#     print("cpus %s" % num_cpus)

#     print("Shuffle configs to create equal computation time chunks ")
#     if num_configs == 0:
#         raise ValueError("No configs to do...")

    
#     seedSeq = SeedSequence(12334567)

#     for sam in range(5):
#         for config in configurations:
#             res = calculate_graphs(config[-1] + '-' + config[0][-1], seedSeq, sam=sam)
#             print(res)
