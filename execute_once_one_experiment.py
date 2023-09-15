from dacite import from_dict
from basic.config import ExecutionConfig
from h_search_unit import h_search_unit
import yaml
from basic.helper import get_dataset_locations
from pathlib import Path
import argparse
import os


def execute_once(dataset_locations, save_folder, experiment_configuration, specific_name=None):
    # with open(f"experiments/{args.experiment}/base_config.yaml", "r") as f:
    #     base_config = yaml.load(f, Loader=yaml.FullLoader)
    config_to_execute = from_dict(data_class=ExecutionConfig, data=experiment_configuration)
    try:
        result = h_search_unit(
            save_folder=save_folder,
            dataset_locations=dataset_locations,
            config_to_execute=config_to_execute,
            specific_name=specific_name
        )
    except Exception as e:
        print(e)
        result = {'score': -1}
    return result

def main(args):
    # If folder does not exist, create it
    os.makedirs(f"execute_once_experiments/scores", exist_ok=True)
    os.makedirs(f"execute_once_experiments/results", exist_ok=True)
    
    # Get the dataset locations
    data_fullpath = Path.absolute(Path(args.data))
    dataset_locations_fullpath = Path.absolute(Path(args.dataset_locations_fullpath))
    dataset_locations = get_dataset_locations(
        data_fullpath=data_fullpath,
        dataset_locations_fullpath=dataset_locations_fullpath
    )
    # Read the hyperparameters search config file
    with open(f"execute_once_experiments/configs/{args.experiment}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
    # Execute the hyperparameters search
    score = execute_once(
        dataset_locations=dataset_locations,
        save_folder=Path.absolute(Path(f"execute_once_experiments/results")),
        experiment_configuration=experiment_config,
        specific_name=args.experiment
    )
    # Save the score in a file
    with open(f"execute_once_experiments/scores/{args.experiment}.yaml", "w") as f:
        yaml.dump(score, f)

# Execute main function
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog="Execute experiments in datasets",
        description="Runs experiments in a dataset with a set of configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_state",
        default=42,
        help="Random state for the experiments",
        type=int,
        required=False,
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
    parser.add_argument(
        "--experiment",
        help="Experiment name",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    print(args)
    main(args=args)