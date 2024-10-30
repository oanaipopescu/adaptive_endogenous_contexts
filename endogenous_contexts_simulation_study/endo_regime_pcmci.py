# License: GNU General Public License v3.0
from __future__ import print_function
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import scipy.stats

from tigramite.pcmci import PCMCI


class EndoRegimePCMCI(PCMCI):
    """
    A specialized PCMCI class for handling endogenous regime changes.
    Extends the base PCMCI functionality to integrate context-specific causal discovery.
    Together with the DataframeRegimeRemoveR, they make the PC-AR algorithm in the paper.
    """
    def __init__(self, dataframe,
                 cond_ind_test,
                 regime_cond_ind=None,
                 pool_disc_cond_ind=None,
                 regime_indicator=None,
                 verbosity=0):
        """
        Initializes the EndoRegimePCMCI instance with necessary parameters and custom testing functionalities.

        Parameters:
        - dataframe: Tigramite DataFrame object.
        - cond_ind_test: The base conditional independence test for continuous data.
        - regime_cond_ind: Conditional independence test for when the regime is in the conditions.
        - pool_disc_cond_ind: Conditional independence test for pooled data when a variable is discrete=.
        - regime_indicator: Context indicator variable index.
        - verbosity: Level of verbosity in output logging.
        """
         # Initialize the base PCMCI class
        PCMCI.__init__(self, dataframe=dataframe, 
                        cond_ind_test=cond_ind_test,
                        verbosity=verbosity)
        # Initialize regime-specific attributes
        self.regime_cond_ind = regime_cond_ind
        self.pool_disc_cond_ind = pool_disc_cond_ind
        self.regime_indicator = regime_indicator
        # Ensure all conditional independence tests refer to the same dataframe
        self.regime_cond_ind.dataframe = self.cond_ind_test.dataframe = self.pool_disc_cond_ind.dataframe = dataframe

    
    def _run_pcalg_test(self, graph, i, abstau, j, S, lagged_parents, max_conds_py,
                        max_conds_px, max_conds_px_lagged, tau_max, alpha_or_thres=None):
        """
        Overrides the PCMCI _run_pcalg_test method to use the adequate test depending on the data to be tested for conditional independence. 
        For more details, see Tigramite documentation.
        
        """
        if lagged_parents is not None:
            conds_y = lagged_parents[j][:max_conds_py]
            if abstau == 0:
                conds_x = lagged_parents[i][:max_conds_px]
            else:
                if max_conds_px_lagged is None:
                    conds_x = lagged_parents[i][:max_conds_px]
                else:
                    conds_x = lagged_parents[i][:max_conds_px_lagged]

        else:
            conds_y = conds_x = []
        # Shift the conditions for X by tau
        conds_x_lagged = [(k, -abstau + k_tau) for k, k_tau in conds_x]

        Z = [node for node in S]
        Z += [node for node in conds_y if
              node != (i, -abstau) and node not in Z]
        Z += [node for node in conds_x_lagged if node not in Z]

        if graph[i,j,abstau] != "" and graph[i,j,abstau][1] == '-':
            val = 1. 
            pval = 0.
            dependent = True
        else:
            X=[(i, -abstau)]
            Y=[(j, 0)]
            Z=Z
            regime_in_Z = False
            for i in range(len(Z)): 
                if Z[i][0] == self.regime_indicator[0]:
                    regime_in_Z = True
            tau_max=tau_max
            # get the arrays
            (array, xyz, XYZ, data_type, nonzero_array, nonzero_xyz, nonzero_XYZ, nonzero_data_type) = self.cond_ind_test._get_array(
                                            X=X, Y=Y, Z=Z, tau_max=tau_max)
            # Modification: select the appropriate conditional independence test based on data type and context presence
            # if the context indicator is in Z, then use a test for continuous variables
            if regime_in_Z:
                val, pval, dependent = self.regime_cond_ind.run_test(X=X, Y=Y,
                                                Z=Z, tau_max=tau_max,
                                                alpha_or_thres=alpha_or_thres,
                                                )
            else:
                # if it is in X or Y, then use a test for mixed data
                if np.any(nonzero_data_type == True) and self.pool_disc_cond_ind is not None:
                    val, pval, dependent = self.pool_disc_cond_ind.run_test(X=X, Y=Y,
                                                    Z=Z, tau_max=tau_max,
                                                    alpha_or_thres=alpha_or_thres,
                                                    # verbosity=self.verbosity
                                                    )
                else:
                    # otherwise, no discrete variable is there, thus use a test for continuous data
                    val, pval, dependent = self.cond_ind_test.run_test(X=X, Y=Y,
                                                        Z=Z, tau_max=tau_max,
                                                        alpha_or_thres=alpha_or_thres,
                                                        # verbosity=self.verbosity
                                                        )
        return val, pval, Z, dependent

    def run_pcalg_non_timeseries_data(self, pc_alpha=0.01,
                  max_conds_dim=None, max_combinations=None, 
                  contemp_collider_rule='majority',
                  conflict_resolution=True, link_assumptions=None):
        """
        Runs the PC algorithm adapted for non-time-series data by setting both tau_min and tau_max to 0. 
        This effectively transforms the algorithm to analyze contemporaneous links only.
        For more details, see Tigramite documentation. 
        """
        results = self.run_pcalg(pc_alpha=pc_alpha, tau_min=0, tau_max=0, 
                    max_conds_dim=max_conds_dim, max_combinations=max_combinations,
                  mode='standard', contemp_collider_rule=contemp_collider_rule,
                  conflict_resolution=conflict_resolution, link_assumptions=link_assumptions)

        # Remove tau-dimension
        old_sepsets = results['sepsets'].copy()
        results['sepsets'] = {}
        for old_sepset in old_sepsets:
           new_sepset = (old_sepset[0][0], old_sepset[1])
           conds = [cond[0] for cond in old_sepsets[old_sepset]]

           results['sepsets'][new_sepset] = conds

        ambiguous_triples = results['ambiguous_triples'].copy()
        results['ambiguous_triples'] = []
        for triple in ambiguous_triples:
           new_triple = (triple[0][0], triple[1], triple[2])

           results['ambiguous_triples'].append(new_triple)
        
        self.pc_results = results
        return results


class MixedTestPCMCI(PCMCI):
    """
    Extends the PCMCI class to accommodate mixed data types (continuous and discrete) for causal discovery.
    Provides methods for running conditional independence tests considering the specific characteristics of the data.
    Used for running the variants PC-M and PC-P in the paper.
    """
    
    def __init__(self, dataframe,
                 cond_ind_test,
                 disc_cond_ind_test=None, 
                 verbosity=0):
        """
        Initializes the MixedTestPCMCI class with specific tests for discrete and continuous data.

        Parameters:
        - dataframe: DataFrame containing the observational data.
        - cond_ind_test: A conditional independence test suited for continuous data.
        - disc_cond_ind_test: A conditional independence test specifically for discrete data.
        - verbosity: Integer level of verbosity to control the output of diagnostic messages.
        """
        # Init base class
        PCMCI.__init__(self, dataframe=dataframe, 
                        cond_ind_test=cond_ind_test,
                        verbosity=verbosity)

        self.disc_cond_ind_test = disc_cond_ind_test

        self.disc_cond_ind_test.dataframe = self.cond_ind_test.dataframe = dataframe

    
    def _run_pcalg_test(self, graph, i, abstau, j, S, lagged_parents, max_conds_py,
                        max_conds_px, max_conds_px_lagged, tau_max, alpha_or_thres=None):
        """
        Custom implementation to handle conditional independence testing by considering both continuous and discrete variables.
        For more details, see Tigramite documentation.
        """
        # Perform independence test adding lagged parents
        if lagged_parents is not None:
            conds_y = lagged_parents[j][:max_conds_py]
            # Get the conditions for node i
            if abstau == 0:
                conds_x = lagged_parents[i][:max_conds_px]
            else:
                if max_conds_px_lagged is None:
                    conds_x = lagged_parents[i][:max_conds_px]
                else:
                    conds_x = lagged_parents[i][:max_conds_px_lagged]

        else:
            conds_y = conds_x = []
        # Shift the conditions for X by tau
        conds_x_lagged = [(k, -abstau + k_tau) for k, k_tau in conds_x]

        Z = [node for node in S]
        Z += [node for node in conds_y if
              node != (i, -abstau) and node not in Z]
        # Remove overlapping nodes between conds_x_lagged and conds_y
        Z += [node for node in conds_x_lagged if node not in Z]

        # If middle mark is '-', then set pval=0
        if graph[i,j,abstau] != "" and graph[i,j,abstau][1] == '-':
            val = 1. 
            pval = 0.
            dependent = True
        else:
            X=[(i, -abstau)]
            Y=[(j, 0)]
            Z=Z
            tau_max=tau_max
            
            (array, xyz, XYZ, data_type, nonzero_array, nonzero_xyz, nonzero_XYZ, nonzero_data_type) = self.cond_ind_test._get_array(
                                            X=X, Y=Y, Z=Z, tau_max=tau_max)
            # Modification: select the appropriate conditional independence test if data is mixed-type or continuous
            if np.any(nonzero_data_type == True) and self.disc_cond_ind_test is not None:
                    val, pval, dependent = self.disc_cond_ind_test.run_test(X=X, Y=Y,
                                                    Z=Z, tau_max=tau_max,
                                                    alpha_or_thres=alpha_or_thres,
                                                    # verbosity=self.verbosity
                                                    )
            else:
                val, pval, dependent = self.cond_ind_test.run_test(X=X, Y=Y,
                                                    Z=Z, tau_max=tau_max,
                                                    alpha_or_thres=alpha_or_thres,
                                                    # verbosity=self.verbosity
                                                    )
        return val, pval, Z, dependent

    def run_pcalg_non_timeseries_data(self, pc_alpha=0.01,
                  max_conds_dim=None, max_combinations=None, 
                  contemp_collider_rule='majority',
                  conflict_resolution=True, link_assumptions=None):
        """
        Runs the PC algorithm adapted for non-time-series data by setting both tau_min and tau_max to 0. 
        This effectively transforms the algorithm to analyze contemporaneous links only.
        For more details, see Tigramite documentation. 
        """
        results = self.run_pcalg(pc_alpha=pc_alpha, tau_min=0, tau_max=0, 
                    max_conds_dim=max_conds_dim, max_combinations=max_combinations,
                  mode='standard', contemp_collider_rule=contemp_collider_rule,
                  conflict_resolution=conflict_resolution, link_assumptions=link_assumptions)

        # Remove tau-dimension
        old_sepsets = results['sepsets'].copy()
        results['sepsets'] = {}
        for old_sepset in old_sepsets:
           new_sepset = (old_sepset[0][0], old_sepset[1])
           conds = [cond[0] for cond in old_sepsets[old_sepset]]

           results['sepsets'][new_sepset] = conds

        ambiguous_triples = results['ambiguous_triples'].copy()
        results['ambiguous_triples'] = []
        for triple in ambiguous_triples:
           new_triple = (triple[0][0], triple[1], triple[2])

           results['ambiguous_triples'].append(new_triple)
        
        self.pc_results = results
        return results
