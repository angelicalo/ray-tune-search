import numpy as np
import random
import torch
import yaml
from pathlib import Path
from basic.config import *


def set_random_state(random_state):
    """
    Set the random state for all the libraries used in the project.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    # Copilot suggestion
    # torch.cuda.manual_seed(random_state)
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(random_state)
    # torch.backends.cudnn.benchmark = False


def get_dataset_locations(data_fullpath: Path, dataset_locations_fullpath: Path):
    # print("TUNE_ORIG_WORKING_DIR:", os.environ.get("TUNE_ORIG_WORKING_DIR"))
    # base_dir = "/home/msc2021-fra/ra264955/new_framework/data/"
    # path = Path("/home/msc2021-fra/ra264955/new_framework/hyperparameters_search/dataset_locations.yaml")
    with Path(dataset_locations_fullpath).open("r") as f:
        base_locations = yaml.load(f, Loader=yaml.CLoader)
    # results = load_yaml("~/new_framework/hyperparameters_search/dataset_locations.yaml")
    for item in base_locations:
        # base_locations[item] =  Path(data_fullpath + base_locations[item])
        # base_locations[item] =  Path(data_fullpath) / base_locations[item]
        base_locations[item] =  Path(data_fullpath) / Path(base_locations[item])
    return base_locations


def process_result(result):
    """
    Process the result of a single experiment run.
    In the original code, the objective is to report the mean and std of the accuracy and f1-score (macro and weighted).
    In this code, at the end of the function, we also report the maximum accuracy found among all the estimators. 

    """
    print(result)
    classifier_results = []
    for report in result['report']:
        classifier_result = {}
        classifier_result['estimator'] = report['estimator']['name']
        classifier_result["accuracy (mean)"] = np.mean(
            # [x["accuracy"] for r in report["results"]["runs"] for x in r["result"]]
            [r['result']["accuracy"] for r in report["results"]["runs"]]
        )
        classifier_result["accuracy (std)"] = np.std(
            # [x["accuracy"] for r in report["results"]["runs"] for x in r["result"]]
            [r['result']["accuracy"] for r in report["results"]["runs"]]
        )
        classifier_result["f1-score macro (mean)"] = np.mean(
            [r['result']["f1 score (macro)"] for r in report["results"]["runs"]]
            # [
            #     x["f1 score (macro)"]
            #     for r in report["results"]["runs"]
            #     for x in r["result"]
            # ]
        )
        classifier_result["f1-score macro (std)"] = np.std(
            [r['result']["f1 score (macro)"] for r in report["results"]["runs"]]
            # [
            #     x["f1 score (macro)"]
            #     for r in report["results"]["runs"]
            #     for x in r["result"]
            # ]
        )
        classifier_result["f1-score weighted (mean)"] = np.mean(
            [r['result']["f1 score (weighted)"] for r in report["results"]["runs"]]
            # [
            #     x["f1 score (weighted)"]
            #     for r in report["results"]["runs"]
            #     for x in r["result"]
            # ]
        )
        classifier_result["f1-score weighted (std)"] = np.std(
            [r['result']["f1 score (weighted)"] for r in report["results"]["runs"]]
            # [
            #     x["f1 score (weighted)"]
            #     for r in report["results"]["runs"]
            #     for x in r["result"]
            # ]
        )
        classifier_results.append(classifier_result)
    
    classifier_results.append(
        {
            "estimator": "MAX(KNN,SVM,RF)",
            "accuracy": np.max([result["accuracy (mean)"] for result in classifier_results])
        }
    )
    return classifier_results

def umap_simple_experiment(umap_config, dataset, random_state):
    """
    Create a config to run a simple experiment with UMAP as a reducer and 3 estimators: RF, SVC and KNN.
    Specifically designed to be used by h_search_unit.py
    """

    # Estimators
    estimator_rf = EstimatorConfig(
        name="randomforest-100",
        algorithm="RandomForest",
        kwargs={"n_estimators": 100},
        num_runs=10,
    )
    estimator_knn = EstimatorConfig(
        name="knn-5",
        algorithm="KNN",
        kwargs={"n_neighbors": 5},
        num_runs=1
    )
    estimator_svm = EstimatorConfig(
        name="svm-rbf-c1.0",
        algorithm="SVM",
        kwargs={"C": 1.0, "kernel": "rbf"},
        num_runs=1
    )
    estimators = [ estimator_rf, estimator_knn, estimator_svm ]
    
    # Datasets
    train_dataset = [dataset+"[train]"]
    test_dataset = [dataset+"[validation]"]
    reducer_dataset = [dataset+"[train]"]
    
    # Reducer
    reducer_umap = ReducerConfig(
        name="umap",
        algorithm="umap",
        kwargs={
            "n_components": umap_config['umap_ncomp'],
            "spread": umap_config['umap_spread'],
            "min_dist": umap_config['umap_mdist'],
            "n_neighbors": umap_config['umap_neigh'],
            "random_state": random_state
        }
    )

    # Version
    version = "1.0"

    # Extra
    extra_info = ExtraConfig(
        in_use_features=[ "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z" ],
        reduce_on="all",
        save_reducer=False,
        scale_on="all"
    )

    # Create config
    config_to_execute = ExecutionConfig(
        version=version,
        estimators=estimators,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        reducer_dataset=reducer_dataset,
        reducer=reducer_umap,
        extra=extra_info,
        scaler=None,
        transforms=[]
    )
    return config_to_execute
