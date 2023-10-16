# Python imports
from typing import Any, Dict, List
import pickle

# Third-party imports
from basic.config import *

# Librep imports
from librep.datasets.multimodal import (
    MultiModalDataset,
    TransformMultiModalDataset,
    WindowedTransform,
)
from basic.utils import multimodal_multi_merge

# Parametric transform
def do_reduce(
    datasets: List[MultiModalDataset],
    reducer_config: ReducerConfig,
    experiment_id: str,
    reduce_on: str = "all",
    suffix: str = "reduced",
    save_reducer: bool = False,
    report_reducer_weight: bool = False,
    save_dir: str = 'reducers/'
) -> List[MultiModalDataset]:
    """Utilitary function to perform dimensionality reduce to a list of
    datasets. The first dataset will be used to fit the reducer. And the
    reducer will be applied to the remaining datasets.

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        List of the datasets to perform the dimensionality reduction.
        The first dataset will be used to fit the reducer. And the
        reducer will be applied to the remaining datasets.
    reducer_config : ReducerConfig
        The reducer configuration, used to instantiate the reducer.
    reduce_on : str, optional
        How reduce will perform, by default "all".
        It can have the following values:
        - "all": the reducer will be applied to the whole dataset.
        - "sensor": the reducer will be applied to each sensor, and then,
            the datasets will be concatenated.
        - "axis": the reducer will be applied to each axis of each sensor,
            and then, the datasets will be concatenated.
    suffix : str, optional
        The new suffix to be appended to the window name, by default "reduced."

    Returns
    -------
    List[MultiModalDataset]
        The list of datasets with the dimensionality reduction applied.
        **Note**: the first will not be transformed (and not returned)
    Raises
    ------
    ValueError
        - If the number of datasets is less than 2.
        - If the reduce_on value is invalid.

    NotImplementedError
        If the reduce_on is not implemented yet.
    """
    # Sanity check
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required to reduce")

    sensor_names = ["accel", "gyro"]
    reducer_needing_y = ['convtae1d', 'lstm', 'convaelstm']
    reducer_reporting_weight = ['convtae1d']

    # Get the reducer kwargs
    kwargs = reducer_config.kwargs or {}
    if reduce_on == "all":
        # Get the reducer class and instantiate it using the kwargs
        reducer = reducers_cls[reducer_config.algorithm](**kwargs)
        # Y initialized
        y_to_use = None
        # Fit the reducer on the first dataset
        # reducer.fit(datasets[0][:][0])
        # WHEN USING AN AUTOENCODER
        if reducer_config.algorithm in reducer_needing_y:
            y_to_use = datasets[0][:][1]
        print(datasets[0][:][0].shape, y_to_use)
        reducer.fit(datasets[0][:][0], y=y_to_use)
        if report_reducer_weight and reducer_config.algorithm in reducer_reporting_weight:
            model_all_params = sum(param.numel() for param in reducer.model.parameters())
            model_all_trainable_params = sum(param.numel() for param in reducer.model.parameters() if param.requires_grad)
            setattr(reducer_config, 'num_params', model_all_params)
            setattr(reducer_config, 'num_trainable_params', model_all_trainable_params)
        if save_reducer:
            filename = save_dir + experiment_id + '.reducer'
            with open(filename, 'wb') as handle:
                pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Instantiate the WindowedTransform with fit_on=None and
        # transform_on="all", i.e. the transform will be applied to
        # whole dataset.
        transform = WindowedTransform(
            transform=reducer,
            fit_on=None,
            transform_on="all",
        )
        # Instantiate the TransformMultiModalDataset with the list of transforms
        # and the new suffix
        transformer = TransformMultiModalDataset(
            transforms=[transform], new_window_name_prefix=suffix
        )
        # Apply the transform to the remaining datasets
        datasets = [transformer(dataset) for dataset in datasets[1:]]
        return datasets

    elif reduce_on == "sensor" or reduce_on == "axis":
        if reduce_on == "axis":
            window_names = datasets[0].window_names
        else:
            window_names = [
                [w for w in datasets[0].window_names if s in w] for s in sensor_names
            ]
            window_names = [w for w in window_names if w]

        window_datasets = []

        # Loop over the windows
        for i, wname in enumerate(window_names):
            # Get the reducer class and instantiate it using the kwargs
            reducer = reducers_cls[reducer_config.algorithm](**kwargs)
            # Y initialized
            y_to_use = None
            # Fit the reducer on the first dataset
            reducer_window = datasets[0].windows(wname)
            # reducer.fit(reducer_window[:][0])
            # WHEN USING AN AUTOENCODER
            if reducer_config.algorithm in reducer_needing_y:
                y_to_use = reducer_window[:][1]
            reducer.fit(reducer_window[:][0], y=y_to_use)
            if save_reducer:
                filename = save_dir + experiment_id + '-' + simplify_wname(wname) + '.reducer'
                with open(filename, 'wb') as handle:
                    pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # Instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            transform = WindowedTransform(
                transform=reducer,
                fit_on=None,
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            # and the new suffix
            transformer = TransformMultiModalDataset(
                transforms=[transform], new_window_name_prefix=f"{suffix}-{i}"
            )
            # Apply the transform to the remaining datasets
            _window_datasets = []
            for dataset in datasets[1:]:
                dset_window = dataset.windows(wname)
                dset_window = transformer(dset_window)
                _window_datasets.append(dset_window)
            window_datasets.append(_window_datasets)

        # Merge dataset windows
        datasets = [
            multimodal_multi_merge([dataset[i] for dataset in window_datasets])
            for i in range(len(window_datasets[0]))
        ]
        return datasets
            

        raise NotImplementedError(f"Reduce_on: {reduce_on} not implemented yet")
    else:
        raise ValueError(
            "Invalid reduce_on value. Must be one of: 'all', 'axis', 'sensor"
        )

def simplify_wname(wname):
    basic_tags = ['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']
    for basic_tag in basic_tags:
        if basic_tag in wname:
            return basic_tag
    return wname
