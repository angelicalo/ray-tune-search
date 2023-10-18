# Python imports
import time
from dataclasses import asdict
from typing import Any, Dict, List

# from config import ExecutionConfig
from basic.config import *

# Filter warnings from UMAP
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# Librep imports
from librep.config.type_definitions import PathLike
from librep.metrics.report import ClassificationReport
from librep.utils.workflow import MultiRunWorkflow, SimpleTrainEvalWorkflow
from basic.utils import catchtime, load_yaml, get_sys_info

# Local imports
from basic.load_datasets import load_datasets
from basic.do_transform import do_transform
from basic.do_reduce import do_reduce
from basic.do_scaling import do_scaling
from basic.do_classification import do_classification
from pathlib import Path

# Function that runs the experiment
def run_basic_experiment(
    dataset_locations: Dict[str, PathLike],
    config_to_execute: ExecutionConfig
) -> dict:
    """This function is the wrapper that runs the experiment.
    The experiment is defined by the config_to_execute parameter,
    which controls the experiment execution.

    This code runs the following steps (in order):
    1. Load the datasets
    2. Perform the non-parametric transformations, if any, using `do_transform`
        function. The transforms are specified by `config_to_execute.transforms`
        which is a list of `TransformConfig` objects.
    3. Perform the parametric transformations, if any, using `do_reduce` function.
        The reducer algorithm and parameters are specified by
        `config_to_execute.reducers` which `ReducerConfig` object.
    4. Perform the scaling, if any, using `do_scaling` function. The scaler
        algorithm and parameters are specified by `config_to_execute.scaler`
        which is a `ScalerConfig` object.
    5. Perform the training and evaluation of the model.
    6. Save the results to a file.

    Parameters
    ----------
    dataset_locations :  Dict[str, PathLike],
        Dictionary with dataset locations. Key is the dataset name and value
        is the path to the dataset.
    experiment_output_file : PathLike
        Path to the file where the results will be saved.
    config_to_execute : ExecutionConfig
        The configuration of the experiment to be executed.

    Returns
    -------
    dict
        Dictionary with the results of the experiment.

    Raises
    ------
    ValueError
        If the reducer is specified but the reducer_dataset is not specified.
    """
    experiment_output_file = Path(experiment_output_file)

    if config_version != config_to_execute.version:
        raise ValueError(
            f"Config version ({config_to_execute.version}) "
            f"does not match the current version ({config_version})"
        )

    # Useful variables
    additional_info = dict()
    start_time = time.time()
    
    # Dictionary to store the datasets
    datasets = dict()

    # ----------- 1. Load the datasets -----------
    with catchtime() as loading_time:
        # Load train dataset (mandatory)
        datasets["train_dataset"] = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.train_dataset,
            features=config_to_execute.extra.in_use_features,
        )
        # Load validation dataset (optional)
        if config_to_execute.validation_dataset:
            datasets["validation_dataset"] = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.validation_dataset,
                features=config_to_execute.extra.in_use_features,
            )

        # Load test dataset (mandatory)
        datasets["test_dataset"] = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.test_dataset,
            features=config_to_execute.extra.in_use_features,
        )

        # Load reducer dataset (optional)
        if config_to_execute.reducer_dataset:
            datasets["reducer_dataset"] = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.reducer_dataset,
                features=config_to_execute.extra.in_use_features,
            )

        if config_to_execute.reducer_validation_dataset:
            datasets["reducer_validation_dataset"] = load_datasets(
                dataset_locations=dataset_locations,
                datasets_to_load=config_to_execute.reducer_validation_dataset,
                features=config_to_execute.extra.in_use_features,
            )

    # Add some meta information
    additional_info["load_time"] = float(loading_time)
    additional_info["train_size"] = len(datasets["train_dataset"])
    additional_info["validation_size"] = (
        len(datasets["validation_dataset"]) if "validation_dataset" in datasets else 0
    )
    additional_info["test_size"] = len(datasets["test_dataset"])
    additional_info["reduce_size"] = (
        len(datasets["reducer_dataset"]) if "reducer_dataset" in datasets else 0
    )
    
    # ----------- 2. Do the non-parametric transform on train, test and reducer datasets ------------

    with catchtime() as transform_time:
        # Is there any transform to do to the datasets?
        if config_to_execute.transforms is not None:
            # Apply the transform
            datasets = do_transform(
                datasets=datasets,
                transform_configs=config_to_execute.transforms,
                keep_suffixes=True,
            )
    additional_info["transform_time"] = float(transform_time)
    
    # ----------- 3. Do the parametric transform on train and test, using the reducer dataset to fit the transform ------------

    with catchtime() as reduce_time:
        # Is there any reducer object and the reducer dataset is specified?
        if config_to_execute.reducer is not None:
            datasets = do_reduce(
                datasets=datasets,
                reducer_config=config_to_execute.reducer,
                reduce_on=config_to_execute.extra.reduce_on,
                use_y=config_to_execute.reducer.use_y,
                report_reducer_weight=config_to_execute.extra.report_reducer_weight
            )
    additional_info["reduce_time"] = float(reduce_time)
    
    # ----------- 4. Do the scaling on train and test, using the train dataset to fit the scaler ------------

    with catchtime() as scaling_time:
        # Is there any scaler to do?
        if config_to_execute.scaler is not None:
            datasets = do_scaling(
                datasets=datasets,
                scaler_config=config_to_execute.scaler,
                scale_on=config_to_execute.extra.scale_on,
            )

    additional_info["scaling_time"] = float(scaling_time)

    # ----------- 5. Do the training, testing and evaluate ------------

    # Create reporter
    reporter = ClassificationReport(
        use_accuracy=True,
        use_f1_score=True,
        use_classification_report=True,
        use_confusion_matrix=True,
        plot_confusion_matrix=False,
    )
    
    # Run all estimators
    all_results = [
        do_classification(
            datasets=datasets,
            estimator_config=estimator_cfg,
            reporter=reporter,
        )
        for estimator_cfg in config_to_execute.estimators
    ]

    # Add some meta information
    end_time = time.time()
    additional_info["total_time"] = end_time - start_time
    additional_info["start_time"] = start_time
    additional_info["end_time"] = end_time
    additional_info["system"] = get_sys_info()

    if config_to_execute.extra.report_reducer_weight:
        additional_info["num_params"] = config_to_execute.reducer.num_params
        additional_info["num_trainable_params"] = config_to_execute.reducer.num_trainable_params
    # ----------- 6. Save results ------------
    values = {
        "experiment": asdict(config_to_execute),
        "report": all_results,
        "additional": additional_info,
    }

    return values