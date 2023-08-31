from hyperparameters_search import hyperparameters_search
from ray import tune
import argparse
from basic.helper import get_dataset_locations
from pathlib import Path
import yaml


# Main function
def main(args):
    # Get the dataset locations
    data_fullpath = Path.absolute(Path(args.data))
    dataset_locations_fullpath = Path.absolute(Path(args.dataset_locations_fullpath))
    dataset_locations = get_dataset_locations(
        data_fullpath=data_fullpath,
        dataset_locations_fullpath=dataset_locations_fullpath
    )

    # Read the hyperparameters search config file
    with open("config_to_evaluate.yaml", "r") as f:
        input_file = yaml.load(f, Loader=yaml.FullLoader)

    # Get the search space, initial params and experiment name from the config file
    search_space = {
        key: getattr(tune, value['tune_function'])(*value['tune_parameters'])
        for key, value in input_file["search_space"].items()
    }
    initial_params = input_file["initial_params"]
    experiment_name = input_file["experiment_name"]
    resources = input_file["resources"]

    # Execute the hyperparameters search
    hyperparameters_search(
        search_space=search_space,
        initial_params=initial_params,
        dataset=args.dataset,
        experiment_name=experiment_name,
        max_concurrent=args.max_concurrent,
        random_state=args.random_state,
        dataset_locations=dataset_locations,
        resources=resources
    )

# Execute main function
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="Execute experiments in datasets",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max_concurrent",
        default=5,
        help="Max number of concurrent executions",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--random_state",
        default=42,
        help="Random state for the experiments",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_locations_fullpath",
        default="basic/dataset_locations.yaml",
        help="Dataset locations full path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--data",
        default="../../data",
        help="Dataset locations full path",
        type=str,
        required=False,
    )
    # parser.add_argument(
    #     "--experiment_name",
    #     default="Test_experiment",
    #     help="Experiment name",
    #     type=str,
    #     required=False,
    # )

    args = parser.parse_args()
    print(args)
    main(args=args)