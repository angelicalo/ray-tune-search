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
        for item in route:
            property_to_modify = getattr(property_to_modify, item)
        property_to_modify = value
    print('EXPERIMENT'*10, basic_experiment_configuration)
    experiment_configuration = from_dict(data_class=ExecutionConfig, data=basic_experiment_configuration)



    try:
        result = h_search_unit(
            # config=config,
            # random_state=random_state,
            # dataset=dataset,
            save_folder=save_folder,
            dataset_locations=dataset_locations,
            experiment_configuration=experiment_configuration
        )
    except Exception as e:
        result = {'score': -1}
    session.report(result)


# TO MODIFY: EXECUTE HYPERPARAMETERS SEARCH
def hyperparameters_search(
        # search_space, initial_params, dataset, experiment_name,
        max_concurrent=5, random_state=42, dataset_locations=None,
        # resources={"cpu": 1, "gpu": 0},
        base_config=None,
        exploration_config=None):
    save_folder = os.path.abspath(f'{experiment_name}/files')
    print(f"Saving results to {save_folder}...")
    os.makedirs(save_folder, exist_ok=True)


    # Set the random state
    set_random_state(random_state)

    print("TUNE_ORIG_WORKING_DIR:", os.environ.get("TUNE_ORIG_WORKING_DIR"))
    print("TUNE_WORKING_DIR:", os.environ.get("TUNE_WORKING_DIR"))
    print("TUNE_RESULT_DIR:", os.environ.get("TUNE_RESULT_DIR"))

    # Get the search space, initial params and experiment name from the config file
    search_space = {
        key: getattr(tune, value['tune_function'])(*value['tune_parameters'])
        for key, value in exploration_config["search_space"].items()
    }
    initial_params = exploration_config["initial_params"]
    experiment_name = exploration_config["experiment_name"]
    resources = exploration_config["resources"]

    hyperopt = HyperOptSearch(points_to_evaluate=initial_params)
    hyperopt = ConcurrencyLimiter(hyperopt, max_concurrent=max_concurrent)

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
        # tune.with_parameters(
        #     my_objective_function,
        #     random_state=random_state,
        #     dataset=dataset,
        #     save_folder=save_folder,
        #     dataset_locations=dataset_locations
        # ),
        trainable=trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=-1,
            scheduler=ASHAScheduler(),
            search_alg=hyperopt,
            time_budget_s=3600*12,
        ),
        run_config=air.RunConfig(
            name=experiment_name,
            stop=ExperimentPlateauStopper(metric="score", std=0.001, top=10, mode="max", patience=0)
        ),
        param_space=search_space
    )

    results = tuner.fit()
    # Save results in a csv file
    results.get_dataframe().to_csv(f"{experiment_name}/data.csv")
    # Report the best result
    best_result = results.get_best_result(metric="score", mode="max")
    to_save = {'config': best_result.config, 'score': float(best_result.metrics['score'])}

    # Save the best result
    with open(f"{experiment_name}/best.yaml", "w") as f:
        yaml.dump(to_save, f)