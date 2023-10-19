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
from basic.helper import process_result


def h_search_unit(
        # config, random_state, dataset,
        dataset_locations, save_folder=None, config_to_execute:ExecutionConfig=None, specific_name=None):
    # Set the random state
    # set_random_state(random_state)
    # Create the experiment config
    # experiment_config = umap_simple_experiment(config, dataset, random_state)
    # Run the experiment
    experiment_result = run_basic_experiment(
        dataset_locations=dataset_locations,
        config_to_execute=config_to_execute
    )
    # Save the results
    if save_folder:
        # Get the number of files in the folder
        item = len(os.listdir(save_folder))
        if specific_name:
            item = specific_name
        # Save the results
        with open(f"{save_folder}/{item}.yaml", "w") as f:
            yaml.dump(experiment_result, f)
    # Return the score
    processed_result = process_result(experiment_result)
    # Get the maximum accuracy between all the estimators
    max_accuracy = processed_result[-1]['accuracy']
    result_object = {'score': max_accuracy}
    criteria = ['accuracy (mean)', 'accuracy (std)', 'f1-score macro (mean)', 'f1-score macro (std)', 'f1-score weighted (mean)', 'f1-score weighted (std)']
    for result in processed_result[:-1]:
        for criterion in criteria:
            if criterion in result:
                result_object[f'{result["estimator"]}-{criterion}'] = result[criterion]

    # score = process_result(experiment_result)[-1]['accuracy']
    
    if 'num_params' in experiment_result['additional']:
        result_object['num_params'] = experiment_result['additional']['num_params']
        # num_params = experiment_result['additional']['num_params']
    if 'num_trainable_params' in experiment_result['additional']:
        result_object['num_trainable_params'] = experiment_result['additional']['num_trainable_params']
        # num_trainable_params = experiment_result['additional']['num_trainable_params']
    # return {'score': max_accuracy, 'num_params': num_params, 'num_trainable_params': num_trainable_params}
    return result_object