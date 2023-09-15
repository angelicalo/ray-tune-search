from hyperparameters_search import hyperparameters_search
from ray import tune
import argparse
from basic.helper import get_dataset_locations
from pathlib import Path
import yaml
from dacite import from_dict
from basic.config import ExecutionConfig



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
    with open(f"experiments/{args.experiment}/exploration_config.yaml", "r") as f:
        exploration_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(f"experiments/{args.experiment}/base_config.yaml", "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    
    experiment_full_path = Path.absolute(Path(f"experiments/{args.experiment}"))
    
    # Execute the hyperparameters search
    hyperparameters_search(
        # search_space=search_space,
        # initial_params=initial_params,
        # dataset=args.dataset,
        # experiment_name=experiment_name,
        max_concurrent=args.max_concurrent,
        random_state=args.random_state,
        dataset_locations=dataset_locations,
        # resources=resources,
        base_config=base_config,
        exploration_config=exploration_config,
        experiment_full_path=experiment_full_path,
        time_budget=args.time_budget
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
        "--time_budget",
        default=3600*12,
        help="Time budget for the experiments",
        type=int,
        required=False,
    )
    # parser.add_argument(
    #     "--dataset",
    #     default=None,
    #     help="Dataset name",
    #     type=str,
    #     required=True,
    # )
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
    parser.add_argument(
        "--experiment",
        help="Experiment folder",
        type=str,
        required=True,
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