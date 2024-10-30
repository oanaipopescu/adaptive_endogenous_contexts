import yaml
import os

def generate_name_from_params(params):
    """
    Generates a hyphen-separated string from the given parameters except the last one.

    Parameters:
    - params (tuple): A tuple containing parameters.

    Returns:
    - str: A string representation of the parameters separated by hyphens.
    """
    # Join all parameters except the last one into a string separated by hyphens
    params_str = '%s-' * (len(params) - 1) % params[:-1]
    params_str = params_str[:-1]

    return params_str

def generate_string_from_params(params):
    """
    Generates a hyphen-separated string from the given parameters.

    Parameters:
    - params (tuple): A tuple containing parameters.

    Returns:
    - str: A string representation of the parameters separated by hyphens.
    """
    # Join all parameters into a string separated by hyphens
    params_str = '%s-' * len(params) % params
    params_str = params_str[:-1]

    return params_str
    

def load_configurations(path):
    """
    Loads configurations from a YAML file.

    Parameters:
    - path (str): The file path to the YAML configuration file.

    Returns:
    - dict: The configuration dictionary.
    """
    # Open the YAML configuration file and load it
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_configurations(config_path):
    """
    Generates a list of configurations based on the parameters specified in a YAML configuration file.

    Parameters:
    - config_path (str): The file path to the YAML configuration file.

    Returns:
    - str: The directory path where results will be stored.
    - list: A list of configuration tuples.
    """
    # Load configurations from the YAML file
    config = load_configurations(config_path)
    configurations = []

    # Compute the base path relative to this script's location
    base_path = os.path.dirname(os.path.abspath(__file__))
     # Compute the absolute path to the directory where results will be stored
    data_directory = os.path.abspath(os.path.join(base_path, '..', config['results_folder']))

    # Generate configurations by iterating over all combinations of parameters
    for sample_size in config['sample_sizes']:
        for N in config['N_values']:
            for density in config['densities']:
                for max_lag in config['max_lags']:
                    for pc_alpha in config['pc_alphas']:
                        for regime_child_known in config['regime_children_known']:
                            for nb_changed_link in config['nb_changed_links']:
                                for nb_regime in config['nb_regimes']:
                                    for cycle_only in config['cycles_only']:
                                        for remove in config['remove_only']:
                                            for cmiknnmixed in config['use_cmiknnmixed']:
                                                for imbalance_factor in config['imbalance_factor']:
                                                    # Prepare the configuration tuple with all parameters
                                                    config_tuple = (N, density, max_lag, pc_alpha, sample_size, regime_child_known,
                                                                    nb_changed_link, nb_regime, config['nb_repeats'], cycle_only,
                                                                    remove, cmiknnmixed, imbalance_factor, data_directory)
                                                    # Generate a suffix for this configuration using the parameters
                                                    suffix = generate_name_from_params(config_tuple)
    
                                                    current_config = (config_tuple, suffix)
                                                    configurations.append(current_config)
    # Return the directory and the list of configurations
    return data_directory, configurations