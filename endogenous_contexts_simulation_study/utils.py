from tigramite.toymodels import structural_causal_processes as toys
import numpy as np


def lin_f(x): return x


def links_to_cyclic_graph(links, tau_max=None):
    """Helper function to convert dictionary of links to graph array format.

    Parameters
    ---------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
        Also format {0:[(0, -1), ...], 1:[...], ...} is allowed.
    tau_max : int or None
        Maximum lag. If None, the maximum lag in links is used.

    Returns
    -------
    graph : array of shape (N, N, tau_max+1)
        Matrix format of graph with 1 for true links and 0 else.
    """
    N = len(links)

    # Get maximum time lag
    min_lag, max_lag = toys._get_minmax_lag(links)

    # Set maximum lag
    if tau_max is None:
        tau_max = max_lag
    else:
        if max_lag > tau_max:
            raise ValueError("tau_max is smaller than maximum lag = %d "
                             "found in links, use tau_max=None or larger "
                             "value" % max_lag)

    graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
    pos_contemp = []
    for j in links.keys():
        for link_props in links[j]:
            if len(link_props) > 2:
                var, lag = link_props[0]
                coeff = link_props[1]
                if coeff != 0.:
                    graph[var, j, abs(lag)] = "-->"
                    if lag == 0:
                        pos_contemp.append([j, var, 0])
            else:
                var, lag = link_props
                graph[var, j, abs(lag)] = "-->"
                if lag == 0:
                    pos_contemp.append([j, var, 0])

    """for j, var, lag in pos_contemp:
        if graph[var, j, lag] != '-->':
            graph[var, j, lag] = "<--" """

    # do post-processing to add bi-directed edges and symmetrize contemp edges
    for j, var, lag in pos_contemp:
        if [var, j, lag] in pos_contemp:
            graph[var, j, lag] = '<->'
            graph[j, var, lag] = '<->'
        else:
            graph[j, var, lag] = '<--'

    return graph


def links_to_cyclic_graph_skeleton(links, tau_max=None):
    """Helper function to convert dictionary of links to graph array format.

    Parameters
    ---------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
        Also format {0:[(0, -1), ...], 1:[...], ...} is allowed.
    tau_max : int or None
        Maximum lag. If None, the maximum lag in links is used.

    Returns
    -------
    graph : array of shape (N, N, tau_max+1)
        Matrix format of graph with 1 for true links and 0 else.
    """

    N = len(links)

    # Get maximum time lag
    min_lag, max_lag = toys._get_minmax_lag(links)

    # Set maximum lag
    if tau_max is None:
        tau_max = max_lag
    else:
        if max_lag > tau_max:
            raise ValueError("tau_max is smaller than maximum lag = %d "
                             "found in links, use tau_max=None or larger "
                             "value" % max_lag)

    graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
    for j in links.keys():
        for link_props in links[j]:
            if len(link_props) > 2:
                var, lag = link_props[0]
                coeff = link_props[1]
                if coeff != 0.:
                    graph[var, j, abs(lag)] = "o-o"
                    if lag == 0:
                        graph[j, var, 0] = "o-o"
            else:
                var, lag = link_props
                graph[var, j, abs(lag)] = "o-o"
                if lag == 0:
                    graph[j, var, 0] = "o-o"

    return graph


