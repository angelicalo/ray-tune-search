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

def my_objective_function(config, random_state, dataset, save_folder, dataset_locations):
    try:
        result = h_search_unit(
            config=config,
            random_state=random_state,
            dataset=dataset,
            save_folder=save_folder,
            dataset_locations=dataset_locations
        )
    except Exception as e:
        result = {'score': -1}
    session.report(result)


# TO MODIFY: EXECUTE HYPERPARAMETERS SEARCH
def hyperparameters_search(search_space, initial_params, dataset, experiment_name, max_concurrent=5, random_state=42, dataset_locations=None):
    save_folder = os.path.abspath(f'{experiment_name}/results/files')
    print(f"Saving results to {save_folder}...")
    os.makedirs(save_folder, exist_ok=True)
    print("TUNE_ORIG_WORKING_DIR:", os.environ.get("TUNE_ORIG_WORKING_DIR"))
    print("TUNE_WORKING_DIR:", os.environ.get("TUNE_WORKING_DIR"))
    print("TUNE_RESULT_DIR:", os.environ.get("TUNE_RESULT_DIR"))

    # Set default values
    if data_fullpath is None:
        data_fullpath = "/home/msc2021-fra/ra264955/new_framework/data/"
    if dataset_locations_fullpath is None:
        dataset_locations_fullpath = "/home/msc2021-fra/ra264955/new_framework/ray-tune-search/basic/dataset_locations.yaml"
    
    # dataset_locations = get_dataset_locations(data_fullpath=data_fullpath, dataset_locations_fullpath=dataset_locations_fullpath)

    hyperopt = HyperOptSearch(points_to_evaluate=initial_params)
    hyperopt = ConcurrencyLimiter(hyperopt, max_concurrent=max_concurrent)

    tuner = tune.Tuner(
        tune.with_parameters(
            my_objective_function,
            random_state=random_state,
            dataset=dataset,
            save_folder=save_folder,
            dataset_locations=dataset_locations
        ),
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=-1,
            scheduler=ASHAScheduler(),
            search_alg=hyperopt
        ),
        run_config=air.RunConfig(
            name=experiment_name,
            stop=ExperimentPlateauStopper(metric="score", std=0.001, top=10, mode="max", patience=0)
        ),
        param_space=search_space
    )

    results = tuner.fit()
    # Save results in a csv file
    results.get_dataframe().to_csv(f"{experiment_name}/results/data.csv")
    # Report the best result
    best_result = results.get_best_result(metric="score", mode="max")
    to_save = {'config': best_result.config, 'score': float(best_result.metrics['score'])}

    # Save the best result
    with open(f"{experiment_name}/results/best.yaml", "w") as f:
        yaml.dump(to_save, f)