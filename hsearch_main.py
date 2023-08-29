from hyperparameters_search import hyperparameters_search
from ray import tune
import argparse
from basic.helper import get_dataset_locations
from pathlib import Path
import yaml

# dataset_locations_fullpath = "../basic/dataset_locations.yaml"

# dataset = 'kuhar.standartized_balanced'
# start_dim = 200
# search_space = {
#     "umap_ncomp": tune.randint(2, 360),
#     "umap_spread": tune.randint(1, 30),
#     "umap_mdist": tune.uniform(0.0, 0.99),
#     "umap_neigh": tune.randint(2, 201)
# }

# initial_params = [
#     {
#         "umap_ncomp": start_dim,
#         "umap_spread": 1,
#         "umap_mdist": 0.1,
#         "umap_neigh": 15
#     }
# ]

# dl28_config = {
#     "data_fullpath": "/home/darlinne.soto/new_framework/data/",
#     "dataset_locations_fullpath": "/home/darlinne.soto/new_framework/ray-tune-search/basic/dataset_locations.yaml"
# }
# dl4_config = {
#     "data_fullpath": "/home/msc2021-fra/ra264955/new_framework/data/",
#     "dataset_locations_fullpath": "/home/msc2021-fra/ra264955/new_framework/ray-tune-search/basic/dataset_locations.yaml"
# }


def main(args):
    data_fullpath = Path.absolute(args.data_fullpath)
    dataset_locations_fullpath = Path.absolute(args.dataset_locations_fullpath)
    dataset_locations = get_dataset_locations(
        data_fullpath=data_fullpath,
        dataset_locations_fullpath=dataset_locations_fullpath
    )
    print(dataset_locations)

    with open("config_to_evaluate.yaml", "r") as f:
        input_file = yaml.load(f, Loader=yaml.FullLoader)
        print(input_file)
    
    search_space = {
        key: getattr(tune, value['tune_function'])(*value['tune_parameters'])
        for key, value in input_file["search_space"].items()
    }
    initial_params = input_file["initial_params"]
    # for key, value in input_file["search_space"].items():
    #     search_space[key] = getattr(tune, value['tune_function'])(*value['tune_parameters'])
    assert 1==0
    hyperparameters_search(
        search_space=search_space,
        initial_params=initial_params,
        dataset=args.dataset,
        experiment_name=args.experiment_name,
        max_concurrent=args.max_concurrent,
        random_state=args.random_state,
        dataset_locations=dataset_locations
    )

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
        default="../basic/dataset_locations.yaml",
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
        "--experiment_name",
        default="Test_experiment",
        help="Experiment name",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    print(args)
    main(args=args)