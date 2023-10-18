from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
import yaml
import os
from h_search_unit import h_search_unit
from ray.air import session
from basic.helper import set_random_state, get_dataset_locations
from basic.config import ExecutionConfig
from copy import deepcopy
from dacite import from_dict
from ray.tune import Callback
import random
from ray.tune.stopper import Stopper
import numpy as np
import pandas as pd
import ray
from pathlib import Path
import functools
import sys
import traceback


class BestResultCallback(Callback):

    def __init__(self, experiment_full_path):
        self.experiment_full_path = experiment_full_path
        self.data = []
        self.errors = []
        self.counter = 0

    def on_trial_result(self, iteration, trials, trial, result, **info):
        new_row = {
            'iteration': iteration,
            'trial_id': trial.trial_id,
            'score': result['score'],
            'config': result['config']
        }
        self.data.append(new_row)
        self.counter += 1
        if self.counter % 50 == 0:
            data_df = pd.DataFrame(self.data)
            data_df.to_csv(f"{self.experiment_full_path}/callback_data.csv", index=False)
        if 'error_message' in result:
            self.errors.append(
                {
                    'trial_id': trial.trial_id,
                    'config': result['config'],
                    'error_type': result['error_type'],
                    'error_message': result['error_message'],
                    'error_traceback': result['error_traceback']
                }
            )
            errors_df = pd.DataFrame(self.errors)
            errors_df.to_csv(f"{self.experiment_full_path}/callback_errors.csv", index=False)

class CustomStopper(Stopper):
    def __init__(
        self,
        metric: str,
        min: int = 1000,
        patience: int = 100,
        experiment_full_path = ''
    ):
        self._metric = metric
        self._patience = patience
        self._iterations = 0
        self._min = min
        self.best_found = 0
        self.counter = 0
        self.trial_ids = []
        # print('CUSTOM STOPPER - init')
        self.experiment_full_path = experiment_full_path
        # print('CUSTOM STOPPER - experiment_full_path', self.experiment_full_path)
        # self.results = []

    def __call__(self, trial_id, result):
        print(f"CUSTOM STOPPER - Trial ids length: {len(self.trial_ids)} patience: {self._patience}")
        # if self.experiment_full_path != '':
            # print(f"CUSTOM STOPPER - Saving results to {self.experiment_full_path}...")
        if trial_id not in self.trial_ids and result[self._metric] > 0:
            self.trial_ids.append(trial_id)
            self._iterations += 1
            # self.results.append(result[self._metric])
            self.counter += 1
            if result[self._metric] > self.best_found:
                self.best_found = result[self._metric]
                self.counter = 0
        # print(f"Iterations: {self._iterations}")
        # print(f"Counter: {self.counter}")
        # print(f'Results: {self.results}')
        return self.stop_all()
    
    def stop_all(self):
        return self._iterations > self._min and self.counter > self._patience

def my_objective_function(
        config,
        # random_state, dataset,
        save_folder, dataset_locations,
        basic_experiment_configuration=None, search_space=None):
    basic_experiment_configuration = deepcopy(basic_experiment_configuration)
    # Update the values for the current experiment
    for key, value in config.items():
        route = search_space[key]['route'].split('/')
        property_to_modify = basic_experiment_configuration
        for key, item in enumerate(route[:-1]):
            # If item is a number, then it is a list
            if item.isdigit():
                item = int(item)
            property_to_modify = property_to_modify[item]
        property_to_modify[route[-1]] = value
    # print('EXPERIMENT'*10, basic_experiment_configuration)
    config_to_execute = from_dict(data_class=ExecutionConfig, data=basic_experiment_configuration)

    try:
        result = h_search_unit(
            # config=config,
            # random_state=random_state,
            # dataset=dataset,
            save_folder=save_folder,
            dataset_locations=dataset_locations,
            config_to_execute=config_to_execute
        )
    except Exception as e:
        print('EXCEPTION FOUND\n', e)
        syserror = sys.exc_info()
        # result = {'score': random.uniform(-20, -10)}
        result = {'score': -0.1, 'num_params': -1, 'num_trainable_params': -1, 'error_type': str(syserror[0]), 'error_message': str(syserror[1]), 'error_traceback': '\n'.join(traceback.format_tb(e.__traceback__))}
        print('EXCEPTION'*10, result)
    session.report(result)


# TO MODIFY: EXECUTE HYPERPARAMETERS SEARCH
def hyperparameters_search(
        # search_space, initial_params, dataset, experiment_name,
        # max_concurrent=5, random_state=42,
        dataset_locations=None,
        # resources={"cpu": 1, "gpu": 0},
        base_config=None,
        exploration_config=None, experiment_full_path=None,
        # time_budget=None,
        experiment_info=None):
    


    # Set the random state
    set_random_state(experiment_info['random_state'])
    
    # Get the search space, initial params and experiment name from the config file
    search_space = {
        key: getattr(tune, value['tune_function'])(*value['tune_parameters'])
        for key, value in exploration_config["search_space"].items()
    }
    initial_params = exploration_config["initial_params"]
    # experiment_name = exploration_config["experiment_name"]
    resources = exploration_config["resources"]
    # ray.init(
    #     num_cpus=resources['cpu'],
    #     num_gpus=resources['gpu'])
    # ray.init(runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}})



    save_folder = os.path.abspath(f'{experiment_full_path}/files')
    # save_folder = os.path.abspath(f'experiments/{experiment}/files')
    print(f"Saving results to {save_folder}...")
    os.makedirs(save_folder, exist_ok=True)

    hyperopt = HyperOptSearch(points_to_evaluate=initial_params)
    hyperopt = ConcurrencyLimiter(hyperopt, max_concurrent=experiment_info['max_concurrent'])

    # Initializing the trainable
    trainable = my_objective_function
    # Setting the parameters for the function
    trainable = tune.with_parameters(
        trainable,
        # random_state=random_state,
        # dataset=dataset,
        save_folder=save_folder,
        dataset_locations=dataset_locations,
        basic_experiment_configuration=base_config,
        search_space=exploration_config['search_space']
    )
    # Allocating the resources needed
    trainable = tune.with_resources(trainable=trainable, resources=resources)
    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=-1,
            scheduler=ASHAScheduler(),
            search_alg=hyperopt,
            time_budget_s=experiment_info['time_budget'],
        ),
        run_config=air.RunConfig(
            name=str(experiment_full_path).split('/')[-1],
            callbacks=[BestResultCallback(experiment_full_path)],
            stop=CustomStopper(metric="score", min=1000, patience=100, experiment_full_path=experiment_full_path)
            # stop=ExperimentPlateauStopper(metric="score", std=0.001, top=10, mode="max", patience=0)
        ),
        param_space=search_space
    )
    if experiment_info['restore']:
        print('Restoring the hyperparameters search...')
        print("TUNE_ORIG_WORKING_DIR:", os.environ.get("TUNE_ORIG_WORKING_DIR"))
        print("TUNE_WORKING_DIR:", os.environ.get("TUNE_WORKING_DIR"))
        print("TUNE_RESULT_DIR:", os.environ.get("TUNE_RESULT_DIR"))
        # Resume experiment with: Tuner.restore(path="/umap_kuhar_p10_2", trainable=...)
        restore_path = f'/home/darlinne.soto/ray_results/{str(experiment_full_path).split("/")[-1]}'
        restore_path = Path(restore_path).as_posix()
        print(restore_path)
        fixed_func = functools.partial(tune.Tuner.restore, path=restore_path, trainable=trainable)
        tuner = fixed_func()
    print('Starting the hyperparameters search...')
    results = tuner.fit()
    print('Finished the hyperparameters search...')
    # Save results in a csv file
    results.get_dataframe().to_csv(f"{experiment_full_path}/data.csv")
    # Report the best result
    best_result = results.get_best_result(metric="score", mode="max")
    to_save = {'config': best_result.config, 'score': float(best_result.metrics['score'])}

    # Save the best result
    with open(f"{experiment_full_path}/best.yaml", "w") as f:
        yaml.dump(to_save, f)