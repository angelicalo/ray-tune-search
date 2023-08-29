# Python imports
import os
from pathlib import Path
import sys
import yaml
# Add upper directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Third-party imports
from basic.config import *
from basic.run_basic_experiment import run_basic_experiment
from basic.helper import umap_simple_experiment, process_result, set_random_state


def h_search_unit(config, random_state, dataset, dataset_locations, save_folder=None):
    # Set the random state
    set_random_state(random_state)
    # Create the experiment config
    experiment_config = umap_simple_experiment(config, dataset, random_state)
    # Run the experiment
    experiment_result = run_basic_experiment(
        dataset_locations=dataset_locations,
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