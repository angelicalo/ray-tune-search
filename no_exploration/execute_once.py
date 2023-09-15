from dacite import from_dict
from basic.config import ExecutionConfig
from h_search_unit import h_search_unit
import yaml
# 320000 = x

def execute_once(dataset_locations, save_folder, basic_experiment_configuration):
    with open(f"experiments/{args.experiment}/base_config.yaml", "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    config_to_execute = from_dict(data_class=ExecutionConfig, data=basic_experiment_configuration)
    try:
        result = h_search_unit(
            # config=config,
            # random_state=random_state,
            # dataset=dataset,
            save_folder=save_folder,
            dataset_locations=dataset_locations,
            config_to_execute=config_to_execute
        )
    except Exception as e:
        print(e)
        result = {'score': -1}
    return result