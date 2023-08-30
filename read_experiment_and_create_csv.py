import os
import pandas as pd
from ray import tune
import yaml
from hyperparameters_search import my_objective_function
# Read ray tune experiment results
with open("config_to_evaluate.yaml", "r") as f:
    input_file = yaml.load(f, Loader=yaml.FullLoader)



experiment_name = "umap_hyperparameters_on_kuhar.standartized_balanced_starting_with_30.60.90...300"
path = os.path.join("../../ray_results", experiment_name)
restored_tuner = tune.Tuner.restore(path=path, trainable=my_objective_function)
result_grid = restored_tuner.get_results()
results_df = pd.DataFrame(result_grid)
# results_df.to_csv('DATA_EXEMPLO.csv')
print(results_df)