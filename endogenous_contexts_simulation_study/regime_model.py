import numpy as np
from utils import lin_f


class RegimeModel:
    """
    A class to sample from a joint structural causal model over different contexts.
    This model is useful for simulating data under different regimes,
    such as normal conditions vs. extreme events.
    """

    def __init__(self, links=dict(), noises=None, seed=None, causal_order=None, extreme_indicator=(3, 0), regime_thresholds=None, imbalance_factor=1., max_lag=5):
        """
        Initialize the RegimeModel.

        Parameters:
        - links (dict): Specifies the causal links for the base graph.
        - noises (list): Initial noise values for the model. If None, they will be generated.
        - seed (int): Random seed for generating reproducible results.
        - causal_order (list): Specifies the order in which variables should be processed.
        - extreme_indicator (tuple): Indicates which variable and at what lag to consider as extreme.
        - regime_thresholds (list): Thresholds that define the boundaries between different regimes.
        - imbalance_factor (float): Factor to adjust the imbalance when computing thresholds.
        - max_lag (int): The maximum lag to consider in the model.
        """
        self.default_regime_index = 0
        self.N = max(sum([list(regime_links.keys()) for regime_links in links.values()], [])) + 1
        self.links = links
        self.noises = noises
        self.seed = seed
        self.causal_order = causal_order

        self.max_lag = max_lag
        self.extreme_indicator = extreme_indicator
        self.thresholds = regime_thresholds
        self.imbalance_factor = imbalance_factor

    def find_regime_index(self, regime_conditions, data, t):
        """
        Determine the context index for a given time step based on context conditions.

        Parameters:
        - regime_conditions (list): List of functions defining the conditions for each regime.
        - data (np.ndarray): Data array.
        - t (int): Time step to evaluate.

        Returns:
        - int: The index of the regime that the data at time t belongs to.
        """
        regime_index = self.default_regime_index  # set default regime
        for ind, regime_condition in enumerate(regime_conditions):
            if regime_condition(data, t):  # if data fulfills regime_condition
                regime_index = ind
                break  # always use first true regime
        return regime_index

    def regime_value(self, val):
        """
        Determine the regime value based on thresholds.

        Parameters:
        - val (float): The value to evaluate.

        Returns:
        - np.ndarray: The regime index based on the value.
        """
        l = ([np.expand_dims((val < self.thresholds[0]).astype(int), 0)] +  
            [(np.expand_dims((self.thresholds[i] >= val).astype(int), 0) + np.expand_dims((val < self.thresholds[i + 1]).astype(int), 0)) for i in range(len(self.thresholds) - 1)] + 
            [np.expand_dims((val >= self.thresholds[-1]).astype(int), 0)])

        r = np.concatenate(l, axis=0).T

        return np.where(r == 1)[0]

    def generate_thresholds(self, data, nb_regimes):
        """
        Generate thresholds for regime transitions based on data quantiles.

        Parameters:
        - data (np.ndarray): Data array to analyze.
        - nb_regimes (int): Number of regimes to consider.

        Returns:
        - np.ndarray: Thresholds that define the regime boundaries.
        """
        quantiles = np.linspace(0, 1, nb_regimes + 1)  # Include 0 to 1 range fully
        adjusted_quantiles = quantiles ** self.imbalance_factor
    
        # Compute thresholds at these adjusted quantiles
        thresholds = np.quantile(data, adjusted_quantiles)[1:-1]  # Ignore the first (0) and last (1) to avoid min and max of data
        
        return thresholds
    
    def generate_data(self, T=1000):
        """
        Generate data according to the structural causal model.
        The idea behind this approach is to generate all the values once 
        for each regime, then generate the regime indicator for each sample.
        In a second pass, we generate all variables again, this time using 
        the correct regime assignment. This way, in the first pass, only 
        the variables which are either independent of the regime indicator 
        or ancestors of the regime indicator are generated. In the second pass
        we generate the descendants of the regime indicator correctly.

        Parameters:
        - T (int): Number of samples to generate.

        Returns:
        - np.ndarray: Simulated data array of shape (T, N).
        """
        nb_regimes = len(self.links)

        random_state = np.random.RandomState(self.seed)
       
        if self.noises is None:  # for now, all regimes have the same noise
            noises = [random_state.randn(T) for j in range(self.N)]
        else:
            noises = self.noises

        data = np.zeros((int(T), self.N), dtype='float32')
        
        # Initialize the data with noise
        for j in range(self.N):
            if j == self.extreme_indicator[0]:
                data[:, j] = 1. * noises[j]
            else:
                data[:, j] = noises[j]

        # Generate data based on the causal links for the regimes 
        
        for t in range(self.max_lag, T):
            for reg in range(nb_regimes):
                
                for j in self.causal_order[reg]:
                    for link_props in self.links[reg][j]:
                        var, lag = link_props[0]
                        coeff = link_props[1]
                        func = link_props[2]
                        
                        data[t, j] += coeff * func(data[t + lag, var])
        
        # Determine the regime thresholds if not provided
        if self.thresholds is None:
            self.thresholds = self.generate_thresholds(data[:, self.extreme_indicator[0]], nb_regimes)
        # Update the extreme indicator variable based on regime thresholds
        for t in range(self.max_lag, T):
            data[t, self.extreme_indicator[0]] = self.regime_value(data[t, self.extreme_indicator[0]])
                        
        regime_var = data[:, self.extreme_indicator[0]].copy()

        # Reset the data to noise and then update based on regime
        for j in range(self.N):
            if j != self.extreme_indicator[0]:
                data[:, j] = noises[j]

        for t in range(self.max_lag, T):
            # find regime that this datapoint belongs to
            regime_index = int(regime_var[t])
            # generate data according to the right regime, and save the regime assignment in data_mask
            for j in self.causal_order[regime_index]:

                for link_props in self.links[regime_index][j]:
                    var, lag = link_props[0]
                    coeff = link_props[1]
                    func = link_props[2]

                    if j != self.extreme_indicator[0]:
                        data[t, j] += coeff * func(data[t + lag, var])

        return data