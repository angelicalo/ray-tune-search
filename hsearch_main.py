from hyperparameters_search import hyperparameters_search
from ray import tune

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

hyperparameters_search(search_space, initial_params, dataset, f"umap_hyperparameters_on_{dataset}_starting_with_{start_dim}", max_concurrent=5, random_state=42)