import yaml
from ray import tune

type_dicts = {
    "randint": tune.randint,
    "uniform": tune.uniform,
    "choice": tune.choice,
    "loguniform": tune.loguniform
}

with open("config_to_evaluate.yaml", "r") as f:
    result = yaml.load(f, Loader=yaml.FullLoader)
    print(result)

search_space = {}
for key, value in result["search_space"].items():
    search_space[key] = getattr(tune, value['tune_function'])(*value['tune_parameters'])
    # search_space[key] = type_dicts[value](result["search_space"][key][0], result["search_space"][key][1])

print(search_space)
print(result["initial_params"])

