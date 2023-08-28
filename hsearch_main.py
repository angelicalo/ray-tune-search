from hyperparameters_search import hyperparameters_search
from ray import tune
import argparse

dataset = 'kuhar.standartized_balanced'
start_dim = 200
search_space = {
    "umap_ncomp": tune.randint(2, 360),
    "umap_spread": tune.randint(1, 30),
    "umap_mdist": tune.uniform(0.0, 0.99),
    "umap_neigh": tune.randint(2, 201)
}

initial_params = [
    {
        "umap_ncomp": start_dim,
        "umap_spread": 1,
        "umap_mdist": 0.1,
        "umap_neigh": 15
    }
]

dl28_config = {
    "data_fullpath": "/home/darlinne.soto/new_framework/data/",
    "dataset_locations_fullpath": "/home/darlinne.soto/new_framework/ray-tune-search/basic/dataset_locations.yaml"
}
dl4_config = {
    "data_fullpath": "/home/msc2021-fra/ra264955/new_framework/data/",
    "dataset_locations_fullpath": "/home/msc2021-fra/ra264955/new_framework/ray-tune-search/basic/dataset_locations.yaml"
}


def main(args):
    hyperparameters_search(
        search_space,
        initial_params,
        dataset,
        f"umap_hyperparameters_on_{dataset}_starting_with_{start_dim}",
        max_concurrent=args.max_concurrent,
        random_state=42,
        data_fullpath=dl28_config["data_fullpath"],
        dataset_locations_fullpath=dl28_config["dataset_locations_fullpath"]
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

    args = parser.parse_args()
    main(args=args)