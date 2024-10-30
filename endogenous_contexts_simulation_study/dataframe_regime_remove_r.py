# License: GNU General Public License v3.0
# The code for the Dataframe object for the PC-AR algorithm
# based on the Tigramite Dataframe object

from tigramite.data_processing import DataFrame
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import math


class DataFrameRegimeRemoveR(DataFrame):
    """
    Extends the Tigramite DataFrame class to handle context indicators.
    This involves selective removal of R if the test should run conditional R=r, 
    such that it is not included in the variables of the conditional test.
    """
    def __init__(self, regime_indicator=(3,-1), **kwargs):
        """
        Initializes the DataFrameRegimeRemoveR class with an option to specify context (regime) indicators.

        Parameters:
        - regime_indicator: Tuple indicating the variable index and lag that denote the context.
        """
        self.regime_indicator = regime_indicator
        DataFrame.__init__(self, **kwargs)

    def construct_array(self, X, Y, Z, tau_max,
                        extraZ=None,
                        mask=None,
                        mask_type=None,
                        data_type=None,
                        return_cleaned_xyz=False,
                        do_checks=True,
                        remove_overlaps=True,
                        cut_off='2xtau_max',
                        verbosity=0):
        """Constructs array from variables X, Y, Z from data.
        Data is of shape (T, N) if analysis_mode == 'single', where T is the
        time series length and N the number of variables, and of (n_ens, T, N)
        if analysis_mode == 'multiple'.

        Parameters
        ----------
        X, Y, Z, extraZ : list of tuples
            For a dependence measure I(X;Y|Z), X, Y, Z can be multivariate of
            the form [(var1, -lag), (var2, -lag), ...]. At least one varlag in Y
            has to be at lag zero. extraZ is only used in CausalEffects class.
        tau_max : int
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X and Z all have the same sample size.
        mask : array-like, optional (default: None)
            Optional mask array, must be of same shape as data.  If it is set,
            then it overrides the self.mask assigned to the dataframe. If it is
            None, then the self.mask is used, if it exists.
        mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        data_type : array-like
            Binary data array of same shape as array which describes whether
            individual samples in a variable (or all samples) are continuous
            or discrete: 0s for continuous variables and 1s for discrete variables.
            If it is set, then it overrides the self.data_type assigned to the dataframe.
        return_cleaned_xyz : bool, optional (default: False)
            Whether to return cleaned X,Y,Z, where possible duplicates are
            removed.
        do_checks : bool, optional (default: True)
            Whether to perform sanity checks on input X,Y,Z
        remove_overlaps : bool, optional (default: True)
            Whether to remove variables from Z/extraZ if they overlap with X or Y.
        cut_off : {'2xtau_max', 'tau_max', 'max_lag', 'max_lag_or_tau_max', 2xtau_max_future}
            If cut_off == '2xtau_max':
                - 2*tau_max samples are cut off at the beginning of the time
                  series ('beginning' here refers to the temporally first
                  time steps). This guarantees that (as long as no mask is
                  used) all MCI tests are conducted on the same samples,
                  independent of X, Y, and Z.

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + 2*tau_max are cut
                  out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off ==  'max_lag':
                - max_lag(X, Y, Z) samples are cut off at the beginning of the
                  time series, where max_lag(X, Y, Z) is the maximum lag of
                  all nodes in X, Y, and Z. These are all samples that can in
                  principle be used.

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + max_lag(X, Y, Z) are
                  cut out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off == 'max_lag_or_tau_max':
                - max(max_lag(X, Y, Z), tau_max) are cut off at the beginning.
                  This may be useful for modeling by comparing multiple
                  models on the same samples.

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + max(max_lag(X, Y,
                  Z), tau_max) are cut out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off == 'tau_max':
                - tau_max samples are cut off at the beginning. This may be
                  useful for modeling by comparing multiple models on the
                  same samples.

                - If at time step t_missing a data value is missing, then the
                  time steps t_missing, ..., t_missing + max(max_lag(X, Y,
                  Z), tau_max) are cut out. The latter part only holds if
                  remove_missing_upto_maxlag=True.

            If cut_off == '2xtau_max_future':
                First, the relevant time steps are determined as for cut_off ==
                'max_lag'. Then, the temporally latest time steps are removed
                such that the same number of time steps remains as there would
                be for cut_off == '2xtau_max'. This may be useful when one is
                mostly interested in the temporally first time steps and would
                like all MCI tests to be performed on the same *number* of
                samples. Note, however, that while the *number* of samples is
                the same for all MCI tests, the samples themselves may be
                different.
        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        array, xyz [,XYZ], data_type : Tuple of data array of shape (dim, n_samples),
            xyz identifier array of shape (dim,) identifying which row in array
            corresponds to X, Y, and Z, and the type mask that indicates which samples
            are continuous or discrete. For example: X = [(0, -1)],
            Y = [(1, 0)], Z = [(1, -1), (0, -2)] yields an array of shape
            (4, n_samples) and xyz is xyz = numpy.array([0,1,2,2]). If
            return_cleaned_xyz is True, also outputs the cleaned XYZ lists.
        """
        if extraZ is None:
            extraZ = []

        if Z is None:
            Z = []

        # If vector-valued variables exist, add them
        def vectorize(varlag):
            vectorized_var = []
            for (var, lag) in varlag:
                for (vector_var, vector_lag) in self.vector_vars[var]:
                    vectorized_var.append((vector_var, vector_lag + lag))
            return vectorized_var

        X = vectorize(X)
        Y = vectorize(Y)
        Z = vectorize(Z)
        extraZ = vectorize(extraZ)

        # Remove duplicates in X, Y, Z, extraZ
        X = list(OrderedDict.fromkeys(X))
        Y = list(OrderedDict.fromkeys(Y))
        Z = list(OrderedDict.fromkeys(Z))
        extraZ = list(OrderedDict.fromkeys(extraZ))

        if remove_overlaps:
            # If a node in Z occurs already in X or Y, remove it from Z
            Z = [node for node in Z if (node not in X) and (node not in Y)]
            extraZ = [node for node in extraZ if (node not in X) and (node not in Y) and (node not in Z)]

        XYZ = X + Y + Z + extraZ
        dim = len(XYZ)

        # Check that all lags are non-positive and indices are in [0,N-1]
        if do_checks:
            self._check_nodes(Y, XYZ, self.Ndata, dim)

        # Use the mask, override if needed
        _mask = mask
        if _mask is None:
            _mask = self.mask
        else:
            _mask = self._check_mask(mask=_mask)

        _data_type = data_type
        if _data_type is None:
            _data_type = self.data_type
        else:
            _data_type = self._check_mask(mask=_data_type, check_data_type=True)

        # Figure out what cut off we will be using
        if cut_off == '2xtau_max':
            max_lag = 2 * tau_max
        elif cut_off == 'max_lag':
            max_lag = abs(np.array(XYZ)[:, 1].min())
        elif cut_off == 'tau_max':
            max_lag = tau_max
        elif cut_off == 'max_lag_or_tau_max':
            max_lag = max(abs(np.array(XYZ)[:, 1].min()), tau_max)
        elif cut_off == '2xtau_max_future':
            ## TODO: CHECK THIS
            max_lag = abs(np.array(XYZ)[:, 1].min())
        else:
            raise ValueError("max_lag must be in {'2xtau_max', 'tau_max', 'max_lag', " \
                             "'max_lag_or_tau_max', '2xtau_max_future'}")

        # Setup XYZ identifier
        index_code = {'x': 0,
                      'y': 1,
                      'z': 2,
                      'e': 3}
        xyz = np.array([index_code[name]
                        for var, name in zip([X, Y, Z, extraZ], ['x', 'y', 'z', 'e'])
                        for _ in var])

        # Run through all datasets and fill a dictionary holding the
        # samples taken from the individual datasets
        samples_datasets = dict()
        data_types = dict()
        self.use_indices_dataset_dict = dict()

        for dataset_key, dataset_data in self.values.items():

            # Apply time offset to the reference points
            ref_points_here = self.reference_points - self.time_offsets[dataset_key]

            # Remove reference points that are out of bounds or are to be
            # excluded given the choice of 'cut_off'
            ref_points_here = ref_points_here[ref_points_here >= max_lag]
            ref_points_here = ref_points_here[ref_points_here < self.T[dataset_key]]

            # Keep track of which reference points would have remained for
            # max_lag == 2*tau_max
            if cut_off == '2xtau_max_future':
                ref_points_here_2_tau_max = self.reference_points - self.time_offsets[dataset_key]
                ref_points_here_2_tau_max = ref_points_here_2_tau_max[ref_points_here_2_tau_max >= 2 * tau_max]
                ref_points_here_2_tau_max = ref_points_here_2_tau_max[
                    ref_points_here_2_tau_max < self.T[dataset_key]]

            # Sort the valid reference points (not needed, but might be useful
            # for detailed debugging)
            ref_points_here = np.sort(ref_points_here)

            # For cut_off == '2xtau_max_future' reduce the samples size the
            # number of samples that would have been obtained for cut_off ==
            # '2xtau_max', removing the temporally latest ones
            if cut_off == '2xtau_max_future':
                n_to_cut_off = len(ref_points_here) - len(ref_points_here_2_tau_max)
                assert n_to_cut_off >= 0
                if n_to_cut_off > 0:
                    ref_points_here = np.sort(ref_points_here)
                    ref_points_here = ref_points_here[:-n_to_cut_off]

            # If no valid reference points are left, continue with the next dataset
            if len(ref_points_here) == 0:
                continue

            if self.bootstrap is not None:

                boot_blocklength = self.bootstrap['boot_blocklength']

                if boot_blocklength == 'cube_root':
                    boot_blocklength = max(1, int(len(ref_points_here) ** (1 / 3)))
                # elif boot_blocklength == 'from_autocorrelation':
                #     boot_blocklength = \
                #         get_block_length(overlapping_residuals.T, xyz=np.zeros(N), mode='confidence')
                elif type(boot_blocklength) is int and boot_blocklength > 0:
                    pass
                else:
                    raise ValueError("boot_blocklength must be integer > 0, 'cube_root', or 'from_autocorrelation'")

                # Chooses THE SAME random seed for every dataset, maybe that's what we want...
                # If the reference points are all the same, this will give the same bootstrap
                # draw. However, if they are NOT the same, they will differ.
                # TODO: Decide whether bootstrap draws should be the same for each dataset and
                # how to achieve that if the reference points differ...
                # random_state = self.bootstrap['random_state']
                random_state = deepcopy(self.bootstrap['random_state'])

                # Determine the number of blocks total, rounding up for non-integer
                # amounts
                n_blks = int(math.ceil(float(len(ref_points_here)) / boot_blocklength))

                if n_blks < 10:
                    raise ValueError("Only %d block(s) for block-sampling," % n_blks +
                                     " choose smaller boot_blocklength!")

                # Get the starting indices for the blocks
                blk_strt = random_state.choice(np.arange(len(ref_points_here) - boot_blocklength), size=n_blks,
                                               replace=True)
                # Get the empty array of block resampled values
                boot_draw = np.zeros(n_blks * boot_blocklength, dtype='int')
                # Fill the array of block resamples
                for i in range(boot_blocklength):
                    boot_draw[i::boot_blocklength] = ref_points_here[blk_strt + i]
                # Cut to proper length
                ref_points_here = boot_draw[:len(ref_points_here)]

            # Construct the data array holding the samples taken from the
            # current dataset
            samples_datasets[dataset_key] = np.zeros((dim, len(ref_points_here)), dtype=dataset_data.dtype)
            for i, (var, lag) in enumerate(XYZ):
                samples_datasets[dataset_key][i, :] = dataset_data[ref_points_here + lag, var]

            # Build the mask array corresponding to this dataset
            if _mask is not None:
                mask_dataset = np.zeros((dim, len(ref_points_here)), dtype='bool')
                for i, (var, lag) in enumerate(XYZ):
                    mask_dataset[i, :] = _mask[dataset_key][ref_points_here + lag, var]

            # Take care of masking
            use_indices_dataset = np.ones(len(ref_points_here), dtype='int')

            # Build the type mask array corresponding to this dataset
            if _data_type is not None:
                data_type_dataset = np.zeros((dim, len(ref_points_here)), dtype='bool')
                for i, (var, lag) in enumerate(XYZ):
                    data_type_dataset[i, :] = _data_type[dataset_key][ref_points_here + lag, var]
                data_types[dataset_key] = data_type_dataset

            # Remove all values that have missing value flag, and optionally as well the time
            # slices that occur up to max_lag after
            if self.missing_flag is not None:
                missing_anywhere = np.array(np.where(np.any(np.isnan(samples_datasets[dataset_key]), axis=0))[0])

                if self.remove_missing_upto_maxlag:
                    idx_to_remove = set(idx + tau for idx in missing_anywhere for tau in range(max_lag + 1))
                else:
                    idx_to_remove = set(idx for idx in missing_anywhere)

                use_indices_dataset[np.array(list(idx_to_remove), dtype='int')] = 0

            if _mask is not None:
                # Remove samples with mask == 1 conditional on which mask_type is used
                # The mask indicates which values of the regime indicator should remain in the dataset
                # MODIFICATION  compared to the original Tigramite version: 
                # if the regime indicator is in the condioning set, then remove it from the variables
                # This avoids testing with a constant (R=r)
                XYZ_list = []
                for var in (X, Y, Z):
                    for i in range(len(var)):
                        XYZ_list.append(var[i])
                if (self.regime_indicator in Z):
                    for idx, cde in index_code.items():
                        slice_select = np.prod(mask_dataset[xyz == cde, :] == False, axis=0)
                        use_indices_dataset *= slice_select
                    new_variables = [i for i in range(len(XYZ_list)) if XYZ_list[i][0] != self.regime_indicator[0]]
                    xyz = xyz[:-1] # remove one from z
                    Z = [z for z in Z if z[0] != self.regime_indicator[0]]
                else:
                    new_variables = np.asarray(list(range(samples_datasets[dataset_key].shape[0])))
            
            # Accordingly update the data array and remove regime indicator from cond. set
            samples_datasets[dataset_key] = samples_datasets[dataset_key][new_variables, :]
            samples_datasets[dataset_key] = samples_datasets[dataset_key][:, use_indices_dataset == 1]
            
            if _data_type is not None:
                data_types[dataset_key] = data_types[dataset_key][new_variables, :]
                data_types[dataset_key] = data_types[dataset_key][:, use_indices_dataset == 1]

        # Concatenate the arrays of all datasets
        array = np.concatenate(tuple(samples_datasets.values()), axis=1)
        if _data_type is not None:
            type_array = np.concatenate(tuple(data_types.values()), axis=1)
        else:
            type_array = None
            
        # Check whether there is any valid sample
        if array.shape[1] == 0:
            raise ValueError("No valid samples")

        # Print information about the constructed array
        if verbosity > 2:
            self.print_array_info(array, X, Y, Z, self.missing_flag, mask_type, type_array, extraZ)

        # Return the array and xyz and optionally (X, Y, Z)
        if return_cleaned_xyz:
            return array, xyz, (X, Y, Z), type_array

        return array, xyz, type_array