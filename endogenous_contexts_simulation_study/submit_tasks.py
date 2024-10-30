import os
from random import shuffle
import subprocess

import argparse

from config_generator import generate_configurations, generate_string_from_params

def update_script(save_folder):
    """
    Generates a SLURM batch script for job submission based on the provided folder.

    Parameters:
    - save_folder (str): The path to the folder where the script will be saved.

    Returns:
    - str: The file name of the script.
    """
    # Extract the last directory name from the save_folder path to use as the job name
    last_name = save_folder.split('/')[-1]
    # Define the full path to the script file
    file_name = save_folder + '/dlr_cluster_submit.sh'
    print('job name', last_name)
    
    script_string = """#!/bin/bash\n#SBATCH --mail-type=FAIL\n#SBATCH --output=runs/slurm-%j.out\n#SBATCH --job-name=""" + last_name + """\n#SBATCH --mail-type=END,FAIL\n#SBATCH --account=USERNAME\n#SBATCH --partition=long\n#SBATCH --ntasks=64\n#SBATCH --mem-per-cpu=32G\n#SBATCH --time=7-00:00:00\n#SBATCH --requeue\nmodule load mpi/intelmpi/2018.4.274\nmodule load miniconda3-4.10.3-gcc-11.2.0-fywepyp\nsource activate VENV\necho $2\nmpirun python3 -u $1\necho "done!"
    """

    # Write the script string to the file
    with open(file_name, 'w') as file:
        file.write(script_string)
    
    return file_name

if __name__ == "__main__":
    # Set up argument parsing for the YAML configuration file path
    parser = argparse.ArgumentParser(description="Run tasks from a YAML file.")
    parser.add_argument('yaml_path', type=str, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Extract the configuration path from the parsed arguments
    config_path = args.yaml_path

    # Number of CPUs used
    num_cpus = 64
    # Whether to overwrite existing results
    overwrite = False

    # Generate configurations from the YAML file
    results_folder, all_configurations = generate_configurations(config_path)
    # Number of repeats for each configuration
    nb_repeats = all_configurations[0][0][8]

    if not os.path.exists(results_folder):
                os.makedirs(results_folder)

    already_there = []
    configurations = []

    # Process each configuration
    for configuration in all_configurations:
        config_params = configuration[0]
        suffix = configuration[1]
        save_folder = config_params[-1] + '/' + suffix

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        current_results_files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]
        
        if suffix not in configurations:
            if (overwrite == False) and (suffix + '.dat' in current_results_files):
                # print("Configuration %s already exists." % conf)
                already_there.append(suffix)
                pass
            else:
                configurations.append(configuration)

    num_configs = len(configurations) 
    
    print("number of todo configs ", num_configs)
    print("number of existing configs ", len(already_there))
    print("cpus %s" % num_cpus)

    print("Shuffle configs to create equal computation time chunks ")
    shuffle(configurations)
    if num_configs == 0:
        raise ValueError("No configs to do...")
    
    # Function to split configurations into chunks
    def split(a, n):
        k, m = len(a) // n, len(a) % n
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    
    # Iterate over configurations to set up and submit jobs
    for config in configurations:
        config_params = config[0]
        config_string = generate_string_from_params(config_params)
        suffix = config[1]
        save_folder = config_params[-1] + '/' + suffix
        
        runs_folder = save_folder + '/runs'
        if not os.path.exists(runs_folder):
            os.makedirs(runs_folder)

        print('saving runs to', runs_folder)
        # Script to be executed in the SLURM job
        job_list = [(conf, i) for i in range(nb_repeats) for conf in config_params]

        use_script = 'compute_regime_graphs_non_timeseries.py'
        # Update the SLURM script with the current configuration
        script_file_name = update_script(config_params[-1])
    
        print('CONF: ', script_file_name)
        # Construct the sbatch command
        submit_string = ['sbatch', '--output', runs_folder + '/slurm-%j.out', '--partition', 'long', '--ntasks', str(num_cpus), script_file_name,
                         use_script + " %d %d %s" % (
                         num_cpus, nb_repeats, config_string)] 
        process = subprocess.Popen(submit_string) 
        # Execute the sbatch command to submit the job
        output = process.communicate()
