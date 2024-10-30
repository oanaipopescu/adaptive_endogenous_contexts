import copy
from collections import OrderedDict

from tigramite.toymodels import structural_causal_processes as toys
import tigramite.plotting as tp
from matplotlib import pyplot as plt

import numpy as np
from utils import lin_f

from regime_model import RegimeModel


def include_regime_children(links, regime_children, regime_indicator):
    """
    Adds regime children to regime links.

    Parameters:
    - links (dict): The causal links for each variable.
    - regime_children (list): List of children variables to be included.
    - regime_indicator (tuple): Index of the regime / context indicator and timelag.

    Returns:
    - dict: Updated links with regime children included.
    """
    for var, tau in regime_children:
        links[var].append((regime_indicator, 100., "reg"))
    return links


def mask_data(data, regime_indicator, nb_regimes, max_lag):
    """
    Creates masks for the data based on the regime value.

    Parameters:
    - data (np.ndarray): The data to be masked.
    - regime_indicator: Index of the regime / context indicator.
    - nb_regimes (int): Number of regimes.
    - max_lag (int): Maximum lag in the model.

    Returns:
    - dict: A dictionary of masks for each regime.
    """
    regime_var, regime_lag = regime_indicator
    data_masks = {i: np.zeros(data.shape, dtype='bool') for i in range(nb_regimes)}
    T, N = data.shape
    for regime in range(nb_regimes):
        for t in range(T):
            if data[t - max_lag, regime_var] != regime:
                data_masks[regime][t - max_lag, :] = True
        
    return data_masks


def check_links(links):
    """
    Checks the links to determine the causal order and if the graph is acyclic.

    Parameters:
    - links (dict): The causal links for each variable.

    Returns:
    - list or None: The causal order if the graph is acyclic; otherwise, None.
    """
    N = len(links.keys())
    max_lag = 0
    contemp_dag = toys._Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: 
                contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N - 1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0:
                raise ValueError("lag must be non-positive int.")
            
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)
    
    if contemp_dag.isCyclic() == 1:
       causal_order = None
    else:
        causal_order = contemp_dag.topologicalSort()
    return causal_order


def clean_regime_children(regime_children):
    """
    Cleans the list of regime children by removing duplicates and those present in every regime.

    Parameters:
    - regime_children (list): A list of regime children across all regimes.

    Returns:
    - list: Cleaned list of regime children.
    """
    # remove regime_children that appear in each regime
    children_to_remove = []
    for child in regime_children[0]:
        is_in_all = True
        for children_i in regime_children:
            if child not in children_i:
                is_in_all = False
                break
        if is_in_all:
            children_to_remove.append(child)

    regime_children = sum(regime_children, [])
    regime_children = [child for child in regime_children if child not in children_to_remove]
    regime_children = list(OrderedDict.fromkeys(regime_children))
    return regime_children

def unionize_links(links, regime_children=None, nb_regimes=2, regime_indicator=None):
    """
    Combines the links from multiple regimes into a union graph.

    Parameters:
    - links (list): List of links for each regime.
    - regime_children (list): List of regime children to include in the union graph.
    - nb_regimes (int): Number of regimes.
    - regime_indicator (tuple):  Index of the regime / context indicator.

    Returns:
    - dict: The union graph combining links from all regimes.
    """
    # links are assumed not to differ in their mechanisms between regimes
    N = len(links[0])
    union_graph = {}
    for var in range(N):
        parents = [links[i][var] for i in range(nb_regimes)]
        parents = sum(parents, [])
        parents = list(OrderedDict.fromkeys(parents))
        union_graph[var] = parents

    # add regime children
    if regime_children is not None:
        union_graph = include_regime_children(copy.deepcopy(union_graph), regime_children, regime_indicator)
    return union_graph

def process_graph_entries(entries):
    """
    Processes entries in the graph to determine the type of connection.

    Parameters:
    - entries (list): List of graph entries to process.

    Returns:
    - str: Processed value for the graph entries.
    """
    # Remove empty strings from the list
    filtered_entries = [entry for entry in entries if entry != '']
    
    # Check if the filtered list is empty
    if len(filtered_entries) == 0:
        return ''
    
    # Check if all entries in the list are the same
    if all(entry == filtered_entries[0] for entry in filtered_entries):
        return filtered_entries[0]
    
    # Check if there are different entries in the list
    if set(filtered_entries) == {'<--', '-->'}:
        return '<-->'

    if any([entry == 'o-o' for entry in filtered_entries]):
        return 'o-o'

    if any([entry == 'x-x' for entry in filtered_entries]):
        return 'x-x'

def unionize_graphs(graphs, nb_regimes):
    """
    Unionizes multiple graphs to combine links from all regimes.

    Parameters:
    - graphs (list): List of graphs for each regime.
    - nb_regimes (int): Number of regimes.

    Returns:
    - np.ndarray: The unionized graph.
    """
    # links are assumed not to differ in their mechanisms between regimes
    N, N, tau_max = graphs[0].shape
    union_graph = np.zeros(graphs[0].shape, dtype='<U3')
    pos_contemp = []
    for i in range(N):
        for j in range(N):
            for lag in range(tau_max):
                graph_entries_list = [graphs[k][i,j,lag] for k in range(nb_regimes)]
                union_graph[i, j, lag] = process_graph_entries(graph_entries_list)
        
    return union_graph

def process_intersection_graph_entries(entries):
    """
    Processes entries in the graph for the PC-B method.

    Parameters:
    - entries (list): List of graph entries to process.

    Returns:
    - str: Processed value for the graph entries.
    """
    if len(entries) == 0:
        return ''

    elif any([entry == '' for entry in entries]):
        return ''

    else:
        # Check if all entries in the list are the same
        if all(entry == entries[0] for entry in entries):
            return entries[0]
        
        # Check if there are different entries in the list
        if set(entries) == {'<--', '-->'}:
            return '<-->'

        if any([entry == 'o-o' for entry in entries]):
            return 'o-o'
        
        if any([entry == 'x-x' for entry in entries]):
            return 'x-x'

        
def intersect_graphs(graphs, regime_indicator):
    """
    Intersects the regime graphs to the pooled data graphs - the implementation of the PC-B method.

    Parameters:
    - graphs (list): List of graphs for each regime.
    - regime_indicator (tuple): Regime indicator.

    Returns:
    - np.ndarray: The intersected graph.
    """
    # links are assumed not to differ in their mechanisms between regimes
    N, N, tau_max = graphs[0].shape
    intersection_graph = np.zeros(graphs[0].shape, dtype='<U3')
    pos_contemp = []
    for i in range(N):
        for j in range(N):
            for lag in range(tau_max):
                if i != regime_indicator[0] and j != regime_indicator[0]:
                    graph_entries_list = [graphs[k][i,j,lag] for k in range(len(graphs))]
                    intersection_graph[i, j, lag] = process_intersection_graph_entries(graph_entries_list)
                if i == regime_indicator[0]:
                    intersection_graph[regime_indicator[0], :, lag] = graphs[1][regime_indicator[0], :, lag]
                if j == regime_indicator[0]:
                    intersection_graph[:, regime_indicator[0], lag] = graphs[1][:, regime_indicator[0], lag]
                    
    return intersection_graph

def generate_regime_model(N, density, child_seeds, nb_regimes, regime_indicator, max_lag, nb_changed_links, 
                          remove_only=False, cycles_only=False,
                          dep_coeffs=[-0.1, -0.2, 0.1, 0.2], auto_coeffs=[-0.1, 0.1]):
    """
    Generates a regime model with the specified parameters.

    Parameters:
    - N (int): Number of variables in the model.
    - density (float): Density of the links in the model.
    - child_seeds (list): List of seeds for generating child links.
    - nb_regimes (int): Number of regimes.
    - regime_indicator (tuple): Regime indicator
    - max_lag (int): Maximum lag in the model.
    - nb_changed_links (int): Number of links to change.
    - remove_only (bool): If True, only remove links; otherwise, add and flip links as well.
    - cycles_only (bool): If True, only generate models with cycles.
    - dep_coeffs (list): Coefficients for dependencies.
    - auto_coeffs (list): Coefficients for self-dependencies. Does not apply to the non-timeseries case,

    Returns:
    - tuple: Contains the base links, joint links, regime children, and causal order.
    """
    L = int(density * (N * (N - 1) / 2))
    
    # generate basic link dictionary that then is modified per regime
    any_causal_order_none = True
    trial_counter = 0
    
    links_base = links_joint = regime_children = causal_order = None

    if max_lag == 0:
        contemp_fraction = 1.
    else:
        contemp_fraction = 0.5

    while trial_counter < 10:
        base_seed = np.random.MT19937(child_seeds[-2])
        noise_seed = np.random.MT19937(child_seeds[-1])
        links_base, _ = toys.generate_structural_causal_process(N=N, L=L, contemp_fraction=contemp_fraction, max_lag=max_lag,
                                                                dependency_coeffs=dep_coeffs,
                                                                auto_coeffs=auto_coeffs,
                                                                noise_seed=noise_seed,
                                                                seed=base_seed)
        # clear all dicts
        # links_joint = {0: links_base}
        links_joint = {}
        regime_children = {}
        # regime_children = {0: []}
        causal_order = []
        regime_children_final = [[]]

        # clean links_i such that regime variable has no children at any lag (also no autocorrelation)
        for var in range(len(links_base)):
            links_base[var] = [item for item in links_base[var] if item[0][0] != regime_indicator[0]]
        
        regime_has_causal_order_none_or_non_unique = False

        for i in range(0, nb_regimes):
            links_i, regime_children_i, causal_order_i = get_links_i(links_base, i, child_seeds, regime_indicator,
                                                                    max_lag, nb_changed_links, 
                                                                    remove_only=remove_only)
            if causal_order_i is None:
                print('regime not valid')
                regime_has_causal_order_none_or_non_unique = True

            if links_i == links_base:
                print('regime equal to base')
                regime_has_causal_order_none_or_non_unique = True

            for key, existing_regimes in links_joint.items():
                if links_i == existing_regimes:
                    regime_has_causal_order_none_or_non_unique = True 

            links_joint[i] = links_i
            regime_children[i] = regime_children_i
            causal_order.append(causal_order_i)
        
        if regime_has_causal_order_none_or_non_unique:
            links_base = links_joint = regime_children = causal_order = None
            trial_counter += 1
            continue

        for k, rc in regime_children.items():
            regime_children_final.append(rc)

        regime_children = clean_regime_children(regime_children_final)
        
        links_with_regime_children = include_regime_children(copy.deepcopy(links_base), regime_children,
                                                                    regime_indicator)

        causal_order_base = check_links(links_with_regime_children)
        
        if causal_order_base is None:
            print('no base without cycles found, redraw base')
            trial_counter += 1
            links_base = links_joint = regime_children = causal_order = None
            continue

        causal_order.insert(0, causal_order_base)

        if cycles_only==False:
            return links_base, links_joint, regime_children, causal_order
        else:
            # check if unionization has a cycle
            union_graph = unionize_links(links_joint, regime_children, nb_regimes=nb_regimes, regime_indicator=regime_indicator)
            causal_order_union = check_links(union_graph)
            if causal_order_union is None:
                return links_base, links_joint, regime_children, causal_order
            else:
                print('union does not have cycles')
                continue

    return links_base, links_joint, regime_children, causal_order

def generate_regime_data(T, links_joint, nb_regimes, child_seeds, regime_indicator, max_lag, causal_order, regime_thresholds=None, imbalance_factor=None):
    """
    Generates data based on the regime model.

    Parameters:
    - T (int): Number of time steps for the data.
    - links_joint (dict): Links for each variable in the joint regime model.
    - nb_regimes (int): Number of regimes.
    - child_seeds (list): List of seeds for generating child links.
    - regime_indicator (tuple): Regime indicator.
    - max_lag (int): Maximum lag in the model.
    - causal_order (list): Causal order for the variables.
    - regime_thresholds (list, optional): Thresholds for defining regimes.
    - imbalance_factor (float, optional): Factor to adjust the imbalance when computing thresholds.

    Returns:
    - tuple: Contains the generated data, masks, and data type.
    """
    regime_toymodel = RegimeModel(links=links_joint, noises=None, seed=np.random.MT19937(child_seeds[-3]), causal_order=causal_order,
                                  regime_thresholds=regime_thresholds, extreme_indicator=regime_indicator, max_lag=max_lag, imbalance_factor=imbalance_factor)

    data = regime_toymodel.generate_data(T=T)
    masks = mask_data(data, regime_indicator, nb_regimes, max_lag)
    data_type = np.zeros(data.shape, dtype=int)
    data_type[:, regime_indicator[0]] = 1
    return data, masks, data_type

def edit_once(links_base, rand_var, system_vars, regime_indicator, max_lag, method, random_state):
    """
    Edits the links base by adding, removing, or flipping links based on a specified method.

    Parameters:
    - links_base (dict): Base links to be edited.
    - rand_var (int): Random variable chosen for editing.
    - system_vars (list): List of system variables.
    - regime_indicator (tuple): Specifies which variable and at what lag is considered as extreme.
    - max_lag (int): Maximum lag in the model.
    - method (str): Method to edit the links ('remove', 'add', 'flip').
    - random_state (np.random.Generator): Random state for generating random values.

    Returns:
    - tuple: Updated links and new regime children.
    """
    links_i = copy.deepcopy(links_base)
    new_regime_children = []

    if method == 'remove':
        rand_parent_ind = random_state.integers(len(links_i[rand_var]))

        # each change to the original link dict indicates an edge from regime variable to the respective variable (R -> (sys, 0))
        # (for now, we assume that such changes only occur contemporaneously)
        links_i[rand_var].pop(rand_parent_ind)
        new_regime_children.append((rand_var, 0))
        
    elif method == 'add':
        rand_parent = system_vars[random_state.integers(len(system_vars))]
        rand_lag = random_state.integers(max_lag + 1)

        if (rand_parent, -rand_lag) not in [item[0] for item in links_i[rand_var]] and not (
                rand_lag == 0 and rand_parent == rand_var):
            rand_coeff = random_state.choice(np.arange(-0.5, 0.5, step=0.1))
            links_i[rand_var].append(((rand_parent, -rand_lag), rand_coeff, lin_f))
            new_regime_children.append((rand_var, 0))
        else:
            return None, None

    elif method == 'flip':  # only sensible for contemp edges
        contemp_parents = [item for item in links_i[rand_var] if (item[0][1] == 0 and item[0][0] != regime_indicator[0])]
        rand_parent = contemp_parents[random_state.integers(len(contemp_parents))]
        # each change to the original link dict indicates an edge from regime variable to the respective variable (R -> (sys, 0))
        # (for now, we assume that such changes only occur contemporaneously)
        links_i[rand_var].remove(rand_parent)
        links_i[rand_parent[0][0]].append(((rand_var, 0), rand_parent[1], rand_parent[2]))
        new_regime_children.append((rand_var, 0))
        new_regime_children.append((rand_parent[0][0], 0))
    
    return links_i, new_regime_children

def add_item(item, items):
    """
    Adds an item to a list if it is not already present.

    Parameters:
    - item: The item to be added.
    - items (list): The list to add the item to.

    Returns:
    - list: The updated list.
    """
    if item not in items:
        items.append(item)

    return items

def get_links_i(links_base, regime, child_seeds, regime_indicator, max_lag,
                nb_changed_links=1, remove_only=False):
    """
    Gets the links for a specific regime by modifying the base links.

    Parameters:
    - links_base (dict): Base links for the model.
    - regime (int): The specific regime to modify links for.
    - child_seeds (list): List of seeds for generating child links.
    - regime_indicator (tuple): Regime indicator
    - max_lag (int): Maximum lag in the model.
    - nb_changed_links (int): Number of links to change.
    - remove_only (bool): If True, only remove links; otherwise, add and flip links as well.

    Returns:
    - tuple: Contains the modified links, regime children, and causal order for the regime.
    """
    # nb_changed_links is the minimum difference between graphs,
    # maximal difference btw. two regime-specific graphs is nb_regimes*nb_changed_links
    trial_seeds = child_seeds[regime].spawn(10000)

    k = 0
    N = len(links_base)

    regime_children = None
    causal_order_i = None

    while k < 5:
        random_state = np.random.default_rng(trial_seeds[k])
        # randomly select a subset of links (of all system variables) that should be removed in each regime
        system_vars = [var for var in range(N) if var != regime_indicator[0]]

        regime_children = []

        changed_edges = 0
        
        trial_counter = 0
        
        links_i = copy.deepcopy(links_base)
        
        while changed_edges < nb_changed_links and trial_counter < 10:
            rand_var = system_vars[random_state.integers(len(system_vars))]
            neighbors_of_rand_var = [i[0][0] for i in links_i[rand_var]]
            
            if remove_only == False:
                augment_methods = ["remove", "add", "flip"]
                
                # only do possible operations 
                if (len(links_i[rand_var]) == 0):
                    augment_methods.remove("remove")
                    
                contemp_parents = [item for item in links_i[rand_var] if (item[0][1] == 0 and item[0][0] != regime_indicator[0])]
                if len(contemp_parents) == 0:
                    augment_methods.remove("flip")
                    
                # now select method
                rand_method = augment_methods[random_state.integers(len(augment_methods))]
            else:
                rand_method = "remove"
                # check if removal is possible:
                if (len(links_i[rand_var]) == 0):
                    # print('not possible', rand_var)
                    trial_counter += 1
                    continue
            
            updated_links, new_regime_children = edit_once(links_i, rand_var, system_vars, regime_indicator, max_lag, rand_method, random_state)

            if updated_links is None:
                # print('no update')
                trial_counter += 1
                continue
                
            elif new_regime_children not in regime_children:
                links_with_regime_children = include_regime_children(copy.deepcopy(updated_links), regime_children,
                                                                regime_indicator)
                causal_order_i = check_links(links_with_regime_children)

                if causal_order_i is None:
                    # print('cyclic')
                    trial_counter += 1
                    continue
                else:
                    links_i = updated_links
                    regime_children += new_regime_children
                    changed_edges += 1
            else:
                trial_counter += 1
                continue

            if changed_edges == nb_changed_links:
                break
        k += 1
        
    return links_i, regime_children, causal_order_i