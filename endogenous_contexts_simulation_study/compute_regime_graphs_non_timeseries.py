import numpy as np
import time
import sys
import socket
import os
import mpi
import copy
import pickle

from matplotlib import pyplot as plt
from numpy.random import SeedSequence


from endo_regime_pcmci import EndoRegimePCMCI, MixedTestPCMCI
import generate_data as gd
from dataframe_regime_remove_r import DataFrameRegimeRemoveR
from config_generator import generate_name_from_params


from tigramite.pcmci import PCMCI
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import data_processing as pp
import tigramite.plotting as tp

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from cmiknnmixed import CMIknnMixed
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.regressionCI import RegressionCI


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
        return None

    # add links of true regime children to regime graphs
    true_regime_links_with_regime_ind = add_children_links(copy.deepcopy(true_regime_links), true_regime_children,
                                                           regime_indicator)
    true_regime_links_with_regime_ind_str ={i: {var: [(val[0], val[1], "func") for val in parents] for var, parents in links.items()} for i, links in true_regime_links_with_regime_ind.items()}
    base_links = {var: [(val[0], val[1], "lin_f") for val in parents] for var, parents in links_base.items()}
    
    true_regime_links_str = {i: {var: [(val[0], val[1], "lin_f") for val in parents] for var, parents in links.items()} for i, links in true_regime_links.items()}
    true_union_links = gd.unionize_links(true_regime_links, true_regime_children, nb_regimes, regime_indicator)
    true_union_links_str = {var: [(val[0], val[1], "lin_f") for val in parents] for var, parents in true_union_links.items()}

    # true_regime_union_graph = gd.unionize_links([toys.links_to_graph(true_regime_links_with_regime_ind_str[regime], tau_max=max_lag) for regime in range(nb_regimes)], nb_regimes) 
    

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
    # generate masks for masking
    ymasks = gd.mask_data(data, regime_indicator, nb_regimes, max_lag)


    disc_cond_ind_test_regimes = disc_cond_ind_test_union = disc_cond_ind_test_ymask = None
    

    dataframes_ymask = [pp.DataFrame(data=data,
                                     mask=ymasks[i],
                                     data_type=data_type,
                                     datatime={0: np.arange(len(data))},
                                     ) for i in range(nb_regimes)]

    dataframes_regimes = [DataFrameRegimeRemoveR(data=data,
                                          mask=masks[i],
                                          data_type=data_type,
                                          datatime={0: np.arange(len(data))},
                                          regime_indicator=regime_indicator) for i in range(nb_regimes)]

    union_dataframe = pp.DataFrame(data=data,
                                   data_type=data_type,
                                   datatime={0: np.arange(len(data))})
    
    if regime_children_known == 'True':  # link_assumptions
        link_assumptions = {j: {(i, -tau): 'o?o' for i in range(N) for tau in range(max_lag+1)} for j in range(N)}
        link_assumptions[regime_indicator[0]] = {}
        for var in range(N):
            for lag in range(max_lag + 1):
                link_assumptions[var][(regime_indicator[0], -lag)] = 'o?o'
                link_assumptions[regime_indicator[0]][(var, -lag)] = 'o?o'
        for var, lag in true_regime_children:
            link_assumptions[var][regime_indicator] = '-->'
            link_assumptions[regime_indicator[0]][(var, 0)] = '<--'


    elif regime_children_known == 'and_parents':
        true_regime_parents = true_regime_links[0][regime_indicator[0]]
        true_regime_parents = [val[0] for val in true_regime_parents]
        link_assumptions = {j: {(i, -tau): 'o?o' for i in range(N) for tau in range(max_lag + 1)} for j in range(N)}
        link_assumptions[regime_indicator[0]] = {}

        for var, lag in true_regime_parents:
            link_assumptions[var][regime_indicator] = '<--'
            link_assumptions[regime_indicator[0]][(var, 0)] = '-->'

        for var, lag in true_regime_children:
            link_assumptions[var][regime_indicator] = '-->'
            link_assumptions[regime_indicator[0]][(var, 0)] = '<--'
    else:  # no link_assumptions necessary
        link_assumptions = None

    # calculate y-masked graphs
    time_start_ymask = time.time()
    try:
        results_ymask = {}
        for regime in range(nb_regimes):
            cond_ind_test_ymask = RobustParCorr(mask_type='y')
            if use_cmiknnmixed:
                disc_cond_ind_test_ymask = CMIknnMixed(sig_samples=100, mask_type='y', knn_type='global', knn=0.2)
            else:
                disc_cond_ind_test_ymask = RegressionCI(mask_type='y')
                
    
            pcmci_ymask = MixedTestPCMCI(dataframe=dataframes_ymask[regime], cond_ind_test=cond_ind_test_ymask, disc_cond_ind_test=disc_cond_ind_test_ymask, verbosity=0)
            
            if max_lag == 0:
                results_ymask[regime] = pcmci_ymask.run_pcalg_non_timeseries_data(pc_alpha=pc_alpha, link_assumptions=link_assumptions)
            else:
                results_ymask[regime] = pcmci_ymask.run_pcalg(tau_min=0, tau_max=max_lag, pc_alpha=pc_alpha, link_assumptions=link_assumptions)
            if plot_data:
                tp.plot_graph(
                    val_matrix=results_ymask[regime]['val_matrix'],
                    graph=results_ymask[regime]['graph'])
                plt.savefig(graphs_path + 'ymask' + str(regime) + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')
            
    except (ValueError, TypeError) as error:
        print("catch no valid samples ymask")
        results_ymask = {}

    time_end_ymask = time.time()

    # calculate regime-graphs (adapted z-mask)
    results_regimes = {}
    time_start_regimes = time.time()
    for regime in range(nb_regimes):
            cond_ind_test = RobustParCorr(mask_type='z')
            regime_cond_ind = RobustParCorr(mask_type='z')
            if use_cmiknnmixed:
                pool_disc_cond_ind = CMIknnMixed(sig_samples=100,  mask_type='z', knn_type='global', knn=0.2)
            else:
                pool_disc_cond_ind = RegressionCI(mask_type='z')    
    
            pcmci_regimes = EndoRegimePCMCI(dataframe=dataframes_regimes[regime], cond_ind_test=cond_ind_test, regime_cond_ind=regime_cond_ind, pool_disc_cond_ind=pool_disc_cond_ind, regime_indicator=regime_indicator, verbosity=0)
            
            if max_lag == 0:
                results_regimes[regime] = pcmci_regimes.run_pcalg_non_timeseries_data(pc_alpha=pc_alpha, link_assumptions=link_assumptions)
            else:
                results_regimes[regime] = pcmci_regimes.run_pcalg(tau_min=0, tau_max=max_lag, pc_alpha=pc_alpha, link_assumptions=link_assumptions)
            
            if plot_data:
                tp.plot_graph(
                    val_matrix=results_regimes[regime]['val_matrix'],
                    graph=results_regimes[regime]['graph'])
                plt.savefig(graphs_path + 'regime' + str(regime) + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')
    
    time_end_regimes = time.time()

    # assemble union graphs of both
    try:
        union_graph_ymask = gd.unionize_graphs([res['graph'] for res in results_ymask.values()], nb_regimes)
    except (IndexError, KeyError) as error:
        print("catch IndexError ymask")
        union_graph_ymask = None

    try:
        union_graph_regimes = gd.unionize_graphs([res['graph'] for res in results_regimes.values()], nb_regimes)
    except (IndexError, KeyError) as error:
        print("catch IndexError regimes")
        union_graph_regimes = None

    time_start_union = time.time()
    
    cond_ind_test_union = RobustParCorr()
    if use_cmiknnmixed:
        disc_cond_ind_test_union = CMIknnMixed(sig_samples=100, knn_type='global', knn=0.2)
    else:
        disc_cond_ind_test_union = RegressionCI()
    
    pcmci_union = MixedTestPCMCI(dataframe=union_dataframe, cond_ind_test=cond_ind_test_union, disc_cond_ind_test=disc_cond_ind_test_union, verbosity=0)
    
    if max_lag == 0:
        results_union = pcmci_union.run_pcalg_non_timeseries_data(pc_alpha=pc_alpha, link_assumptions=link_assumptions)
    else:
        results_union = pcmci_union.run_pcalg(tau_min=0, tau_max=max_lag, pc_alpha=pc_alpha, link_assumptions=link_assumptions)
    if plot_data:
        tp.plot_graph(
            val_matrix=results_union['val_matrix'],
            graph=results_union['graph'])
        plt.savefig(graphs_path + 'found_pcmci_all' + str(regime) + '_trial_' + str(sam) + 'use_cmiknnmixed_' + str(use_cmiknnmixed) + '_cycles_only_' + str(cycles_only) + '.png')
        
    time_end_union = time.time()

    computation_time_ymask = time_end_ymask - time_start_ymask
    computation_time_regimes = time_end_regimes - time_start_regimes
    computation_time_union = time_end_union - time_start_union

    return {
        'base_links': base_links,
        'causal_order': causal_order,
        'true_regime_links': true_regime_links_str,
        'true_regime_children': true_regime_children,
        'true_regime_links_with_regime_ind': true_regime_links_with_regime_ind_str,
        'true_union_links': true_union_links_str,
        'regime_indicator': regime_indicator,

        'computation_time_ymask': computation_time_ymask,
        'computation_time_regimes': computation_time_regimes,
        'computation_time_pcmci': computation_time_union,

        'graphs_ymask': {regime: res['graph'] for regime, res in results_ymask.items()},
        'graphs_regimes': {regime: res['graph'] for regime, res in results_regimes.items()},

        'union_graph_ymask': union_graph_ymask,
        'union_graph_regimes': union_graph_regimes,
        'union_graph_pcmci': results_union['graph'],
    }

######## IMPORTANT: if using sequential, comment these out

def process_chunks(job_id, chunk, seed):
    results = {}
    num_here = len(chunk)
    model_seeds = seed.spawn(num_here)
    time_start_process = time.time()
    for isam, config_sam in enumerate(chunk):
        results[config_sam] = calculate_graphs(config_sam, model_seeds[isam], plot_data=plot_data, sam=job_id)
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
            file_name_cleaned = file_name.replace("'", "").replace('"', '') + '.dat'
            print('writing... ', file_name_cleaned)
            file = open(file_name_cleaned, 'wb')
            pickle.dump(tmp[conf_sam], file, protocol=-1)
            file.close()
            id_num += 1

    time_end = time.time()
    print('Run time in hours ', (time_end - time_start) / 3600.)


mpi.run(verbose=False)

##### IMPORTANT: if using sequential, comment this in!

# if __name__=='__main__':

#     parser = argparse.ArgumentParser(description="Run tasks from a YAML file.")
#     parser.add_argument('yaml_path', type=str, help='Path to the YAML configuration file.')
#     args = parser.parse_args()
    
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
    
#     for sam in range(nb_repeats): # change here to number 
#         for config in configurations:
#             res = calculate_graphs(config[-1] + '-' + config[0][-1], seedSeq, sam=sam)
#             print(res)
