# Python imports
import os
from pathlib import Path
import sys
import yaml
# Add upper directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Third-party imports
from config import *
from run_basic_experiment import run_basic_experiment
from helper import umap_simple_experiment, process_result, get_dataset_locations, set_random_state


def h_search_unit(config, random_state, dataset, data_fullpath, dataset_locations_fullpath, save_folder=None):
    # Set the random state
    set_random_state(random_state)
    # Create the experiment config
    experiment_config = umap_simple_experiment(config, dataset, random_state)
    # Run the experiment
    experiment_result = run_basic_experiment(
        dataset_locations=get_dataset_locations(data_fullpath, dataset_locations_fullpath),
        config_to_execute=experiment_config
    )
    # Save the results
    if save_folder:
        # Get the number of files in the folder
        item = len(os.listdir(save_folder))
        # Save the results
        with open(f"{save_folder}/{item}.yaml", "w") as f:
            yaml.dump(experiment_result, f)
    # Return the score
    score = process_result(experiment_result)[-1]['accuracy']
    return {'score': score}