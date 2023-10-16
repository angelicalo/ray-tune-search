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
    # experiment_output_file = Path(experiment_output_file)

    # Useful variables
    additional_info = dict()
    start_time = time.time()

    # ----------- 1. Load the datasets -----------

    # Load train dataset
    train_dset = load_datasets(
        dataset_locations=dataset_locations,
        datasets_to_load=config_to_execute.train_dataset,
        features=config_to_execute.extra.in_use_features,
    )
    # Load test dataset
    test_dset = load_datasets(
        dataset_locations=dataset_locations,
        datasets_to_load=config_to_execute.test_dataset,
        features=config_to_execute.extra.in_use_features,
    )
    # If there is any reducer dataset speficied, load reducer
    if config_to_execute.reducer_dataset:
        reducer_dset = load_datasets(
            dataset_locations=dataset_locations,
            datasets_to_load=config_to_execute.reducer_dataset,
            features=config_to_execute.extra.in_use_features,
        )
    else:
        reducer_dset = None
    
    # ----------- 2. Do the non-parametric transform on train, test and reducer datasets ------------
    
    # Is there any transform to do?
    if config_to_execute.transforms is not None:
        # If there is a reducer dataset, do the transform on all of them
        if reducer_dset is not None:
            train_dset, test_dset, reducer_dset = do_transform(
                datasets=[train_dset, test_dset, reducer_dset],
                transform_configs=config_to_execute.transforms,
                keep_suffixes=True,
            )
        # If there is no reducer dataset, do the transform only on train and test
        else:
            train_dset, test_dset = do_transform(
                datasets=[train_dset, test_dset],
                transform_configs=config_to_execute.transforms,
                keep_suffixes=True,
            )
    
    # ----------- 3. Do the parametric transform on train and test, using the reducer dataset to fit the transform ------------

    # Is there any reducer to do?
    if config_to_execute.reducer is not None and reducer_dset is not None:
        train_dset, test_dset = do_reduce(
            datasets=[reducer_dset, train_dset, test_dset],
            reducer_config=config_to_execute.reducer,
            reduce_on=config_to_execute.extra.reduce_on,
            save_reducer=config_to_execute.extra.save_reducer,
            report_reducer_weight=config_to_execute.extra.report_reducer_weight,
            experiment_id=0,
            save_dir="reducers/"
        )
    
    # ----------- 4. Do the scaling on train and test, using the train dataset to fit the scaler ------------

    # Is there any scaler to do?
    if config_to_execute.scaler is not None:
        train_dset, test_dset = do_scaling(
            datasets=[train_dset, test_dset],
            scaler_config=config_to_execute.scaler,
            scale_on=config_to_execute.extra.scale_on,
        )

    # ----------- 5. Do the training, testing and evaluate ------------

    # Create reporter
    reporter = ClassificationReport(
        use_accuracy=True,
        use_f1_score=True,
        use_classification_report=True,
        use_confusion_matrix=True,
        plot_confusion_matrix=False
    )

    all_results = []

    # Create Simple Workflow
    for estimator_cfg in config_to_execute.estimators:
        results = dict()

        workflow = SimpleTrainEvalWorkflow(
            estimator=estimator_cls[estimator_cfg.algorithm],
            estimator_creation_kwags=estimator_cfg.kwargs or {},
            do_not_instantiate=False,
            do_fit=True,
            evaluator=reporter,
        )

        # Create a multi execution workflow
        runner = MultiRunWorkflow(workflow=workflow, num_runs=estimator_cfg.num_runs)
        with catchtime() as classification_time:
            results["results"] = runner(train_dset, test_dset)

        results["classification_time"] = float(classification_time)
        results["estimator"] = asdict(estimator_cfg)
        all_results.append(results)

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