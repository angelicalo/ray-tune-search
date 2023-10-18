from dataclasses import asdict
from typing import Dict
from librep.datasets.multimodal import MultiModalDataset
from basic.config import EstimatorConfig, estimator_cls
from librep.utils.workflow import MultiRunWorkflow, SimpleTrainEvalWorkflowMultiModal
from librep.base.evaluators import SupervisedEvaluator
from utils import catchtime

def do_classification(
    datasets: Dict[str, MultiModalDataset],
    estimator_config: EstimatorConfig,
    reporter: SupervisedEvaluator,
    train_dataset_name="train_dataset",
    validation_dataset_name="validation_dataset",
    test_dataset_name="test_dataset",
) -> dict:
    """Utilitary function to perform classification to a list of datasets.

    Parameters
    ----------
    datasets : Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective dataset.
    estimator_config : EstimatorConfig
        The estimator configuration, used to instantiate the estimator.
    reporter : SupervisedEvaluator
        The reporter object, used to evaluate the model.

    Returns
    -------
    dict
        Dictionary with the results of the experiment.
    """
    results = dict()
    
    # Get the estimator class and instantiate it using the kwargs
    estimator = estimator_cls[estimator_config.algorithm](
        **(estimator_config.kwargs or {})
    )
    # Instantiate the SimpleTrainEvalWorkflowMultiModal
    workflow = SimpleTrainEvalWorkflowMultiModal(
        estimator=estimator,
        do_fit=True,
        evaluator=reporter,
    )
    # Instantiate the MultiRunWorkflow
    runner = MultiRunWorkflow(workflow=workflow, num_runs=estimator_config.num_runs)
    # Run the workflow
    with catchtime() as classification_time:
        results["results"] = runner(
            datasets[train_dataset_name],
            datasets[validation_dataset_name] if validation_dataset_name in datasets else None,
            datasets[test_dataset_name],
        )
        
    results["classification_time"] = float(classification_time)
    results["estimator"] = asdict(estimator_config)

    return results