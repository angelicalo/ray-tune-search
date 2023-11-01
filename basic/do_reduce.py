# Python imports
from typing import Any, Dict, List
from collections import defaultdict

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
    datasets: Dict[str, MultiModalDataset],
    reducer_config: ReducerConfig,
    reducer_dataset_name="reducer_dataset",
    reducer_validation_dataset_name="reducer_validation_dataset",
    reduce_on: str = "all",
    suffix: str = "reduced",
    use_y: bool = False,
    apply_only_in: List[str] = None,
    sensor_names: List[str] = ("accel", "gyro"),
    report_reducer_weight: bool = False,
    model_with_weights: List[str] = ['convtae1d'],
) -> Dict[str, MultiModalDataset]:
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
    #  ----- Sanity check -----

    # The reducer_dataset_name must be in the datasets
    if reducer_dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{reducer_dataset_name}' not found. "
            + f"Maybe you forgot to load it in your configuration file? "
            + f"Check if any 'reducer_dataset' is defined in your configuration file."
        )

    if apply_only_in is not None:
        for dset_name in apply_only_in:
            if dset_name not in datasets:
                raise ValueError(
                    f"Dataset '{dset_name}' not found. "
                    + f"Maybe you forgot to load it in your configuration file?"
                )
    else:
        apply_only_in = list(datasets.keys())

    # Remove reducer_dataset_name and reducer_validation_dataset_name from apply_only_in
    apply_only_in = [
        dset_name
        for dset_name in apply_only_in
        if dset_name not in (reducer_dataset_name, reducer_validation_dataset_name)
    ]

    # Get the reducer kwargs
    kwargs = reducer_config.kwargs or {}

    # Output datasets
    new_datasets = {k: v for k, v in datasets.items()}

    # If reduce on is "all", fit the reducer on the first dataset and
    # apply the reducer to the remaining datasets
    if reduce_on == "all":
        # Get the reducer class and instantiate it using the kwargs
        reducer = reducers_cls[reducer_config.algorithm](**kwargs)
        # Fit the reducer on the reducer_dataset_name
        fit_dsets = {
            "X": datasets[reducer_dataset_name][:][0],
        }
        # If use_y is True, train the reducer with X and y
        if use_y:
            fit_dsets["y"] = datasets[reducer_dataset_name][:][1]
        # If the reducer_validation_dataset_name is in the datasets, use it
        if reducer_validation_dataset_name in datasets:
            fit_dsets["X_val"] = datasets[reducer_validation_dataset_name][:][0]
        # If the reducer_validation_dataset_name is in the datasets and use_y is True,
        # use it
        if reducer_validation_dataset_name in datasets and use_y:
            fit_dsets["y_val"] = datasets[reducer_validation_dataset_name][:][1]

        # Fit the reducer the datasets specified in fit_dsets
        reducer.fit(**fit_dsets)

        try:
            model_all_params = sum(param.numel() for param in reducer.model.parameters())
            model_all_trainable_params = sum(param.numel() for param in reducer.model.parameters() if param.requires_grad)    
        except:
            model_all_params = -1
            model_all_trainable_params = -1
        setattr(reducer_config, 'num_params', model_all_params)
        setattr(reducer_config, 'num_trainable_params', model_all_trainable_params)
            
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
        new_datasets.update(
            {dset_name: transformer(datasets[dset_name]) for dset_name in apply_only_in}
        )

    # If reduce on is "sensor" or "axis", fit the reducer on each sensor
    # and if reduce on is "axis", fit the reducer on each axis of each sensor
    elif reduce_on == "sensor" or reduce_on == "axis":
        if reduce_on == "axis":
            window_names = datasets["reducer_dataset"].window_names
        else:
            window_names = [
                [w for w in datasets["reducer_dataset"].window_names if s in w]
                for s in sensor_names
            ]
            window_names = [w for w in window_names if w]

        window_datasets = defaultdict(list)

        # Loop over the windows (accel, gyro, for "sensor"; (accel-x, accel-y, accel-z, gyro-x, gyro-y, gyro-z, for "axis")
        for i, wname in enumerate(window_names):
            # Get the reducer class and instantiate it using the kwargs
            reducer = reducers_cls[reducer_config.algorithm](**kwargs)
            # Fit the reducer on the first dataset
            reducer_window = datasets["reducer_dataset"].windows(wname)
            # Fit the reducer on the reducer_dataset_name
            fit_dsets = {
                "X": datasets[reducer_dataset_name].windows(wname)[:][0],
            }
            # If use_y is True, train the reducer with X and y
            if use_y:
                fit_dsets["y"] = datasets[reducer_dataset_name].windows(wname)[:][1]
            # If the reducer_validation_dataset_name is in the datasets, use it
            if reducer_validation_dataset_name in datasets:
                fit_dsets["X_val"] = datasets[reducer_validation_dataset_name].windows(
                    wname
                )[:][0]
            # If the reducer_validation_dataset_name is in the datasets and use_y is True,
            # use it
            if reducer_validation_dataset_name in datasets and use_y:
                fit_dsets["y_val"] = datasets[reducer_validation_dataset_name].windows(
                    wname
                )[:][1]

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

            # Apply the transform on the same window of each dataset
            # in apply_only_in
            for dataset_name in apply_only_in:
                dset_window = datasets[dataset_name].windows(wname)
                dset_window = transformer(dset_window)
                window_datasets[dataset_name].append(dset_window)

        # Merge dataset windows
        new_datasets.update(
            {
                dset_name: multimodal_multi_merge(window_datasets[dset_name])
                for dset_name in apply_only_in
            }
        )

    else:
        raise ValueError(
            "Invalid reduce_on value. Must be one of: 'all', 'axis', 'sensor"
        )

    # if reducer_dataset_name in datasets:
    #     new_datasets[reducer_dataset_name] = datasets[reducer_dataset_name]
    # if reducer_validation_dataset_name in datasets:
    #     new_datasets[reducer_validation_dataset_name] = datasets[
    #         reducer_validation_dataset_name
    #     ]
    return new_datasets
    

    # # Get the reducer kwargs
    # kwargs = reducer_config.kwargs or {}

    # # Output datasets
    # new_datasets = {k: v for k, v in datasets.items()}
    
    # # If reduce on is "all", fit the reducer on the first dataset and
    # # apply the reducer to the remaining datasets
    # if reduce_on == "all":
    #     # Get the reducer class and instantiate it using the kwargs
    #     reducer = reducers_cls[reducer_config.algorithm](**kwargs)
    #     # Fit the reducer on the reducer_dataset_name
    #     fit_dsets = {
    #         "X": datasets[reducer_dataset_name][:][0],
    #     }
    #     # If use_y is True, train the reducer with X and y
    #     if use_y:
    #         fit_dsets["y"] = datasets[reducer_dataset_name][:][1]
    #     # If the reducer_validation_dataset_name is in the datasets, use it
    #     if reducer_validation_dataset_name in datasets:
    #         fit_dsets["X_val"] = datasets[reducer_validation_dataset_name][:][0]
    #     # If the reducer_validation_dataset_name is in the datasets and use_y is True,
    #     # use it
    #     if reducer_validation_dataset_name in datasets and use_y:
    #         fit_dsets["y_val"] = datasets[reducer_validation_dataset_name][:][1]
        
    #     # Fit the reducer the datasets specified in fit_dsets
    #     reducer.fit(**fit_dsets)

    #     # Instantiate the WindowedTransform with fit_on=None and
    #     # transform_on="all", i.e. the transform will be applied to
    #     # whole dataset.
    #     transform = WindowedTransform(
    #         transform=reducer,
    #         fit_on=None,
    #         transform_on="all",
    #     )
    #     # Instantiate the TransformMultiModalDataset with the list of transforms
    #     # and the new suffix
    #     transformer = TransformMultiModalDataset(
    #         transforms=[transform], new_window_name_prefix=suffix
    #     )
    #     # Apply the transform to the remaining datasets
    #     new_datasets.update(
    #         {dset_name: transformer(datasets[dset_name]) for dset_name in apply_only_in}
    #     )

    #     # Y initialized
    #     y_to_use = None
    #     # Fit the reducer on the first dataset
    #     # reducer.fit(datasets[0][:][0])
    #     # WHEN USING AN AUTOENCODER
    #     if reducer_config.algorithm in reducer_needing_y:
    #         y_to_use = datasets[0][:][1]
    #     # print(datasets[0][:][0].shape, y_to_use)
    #     reducer.fit(datasets[0][:][0], y=y_to_use)
    #     if report_reducer_weight and reducer_config.algorithm in reducer_reporting_weight:
    #         model_all_params = sum(param.numel() for param in reducer.model.parameters())
    #         model_all_trainable_params = sum(param.numel() for param in reducer.model.parameters() if param.requires_grad)
    #         # print('Model params:', model_all_params)
    #         # print('Model trainable params:', model_all_trainable_params)
    #         setattr(reducer_config, 'num_params', model_all_params)
    #         # print('Reducer config', reducer_config)
    #         # print('Reducer config num_params', reducer_config.num_params)
    #         setattr(reducer_config, 'num_trainable_params', model_all_trainable_params)
    #     if save_reducer:
    #         filename = save_dir + experiment_id + '.reducer'
    #         with open(filename, 'wb') as handle:
    #             pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    #     # Instantiate the WindowedTransform with fit_on=None and
    #     # transform_on="all", i.e. the transform will be applied to
    #     # whole dataset.
    #     transform = WindowedTransform(
    #         transform=reducer,
    #         fit_on=None,
    #         transform_on="all",
    #     )
    #     # Instantiate the TransformMultiModalDataset with the list of transforms
    #     # and the new suffix
    #     transformer = TransformMultiModalDataset(
    #         transforms=[transform], new_window_name_prefix=suffix
    #     )
    #     # Apply the transform to the remaining datasets
    #     datasets = [transformer(dataset) for dataset in datasets[1:]]
    #     return datasets

    # elif reduce_on == "sensor" or reduce_on == "axis":
    #     if reduce_on == "axis":
    #         window_names = datasets[0].window_names
    #     else:
    #         window_names = [
    #             [w for w in datasets[0].window_names if s in w] for s in sensor_names
    #         ]
    #         window_names = [w for w in window_names if w]

    #     window_datasets = []

    #     # Loop over the windows
    #     for i, wname in enumerate(window_names):
    #         # Get the reducer class and instantiate it using the kwargs
    #         reducer = reducers_cls[reducer_config.algorithm](**kwargs)
    #         # Y initialized
    #         y_to_use = None
    #         # Fit the reducer on the first dataset
    #         reducer_window = datasets[0].windows(wname)
    #         # reducer.fit(reducer_window[:][0])
    #         # WHEN USING AN AUTOENCODER
    #         if reducer_config.algorithm in reducer_needing_y:
    #             y_to_use = reducer_window[:][1]
    #         reducer.fit(reducer_window[:][0], y=y_to_use)
    #         if save_reducer:
    #             filename = save_dir + experiment_id + '-' + simplify_wname(wname) + '.reducer'
    #             with open(filename, 'wb') as handle:
    #                 pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         # Instantiate the WindowedTransform with fit_on=None and
    #         # transform_on="all", i.e. the transform will be applied to
    #         # whole dataset.
    #         transform = WindowedTransform(
    #             transform=reducer,
    #             fit_on=None,
    #             transform_on="all",
    #         )
    #         # Instantiate the TransformMultiModalDataset with the list of transforms
    #         # and the new suffix
    #         transformer = TransformMultiModalDataset(
    #             transforms=[transform], new_window_name_prefix=f"{suffix}-{i}"
    #         )
    #         # Apply the transform to the remaining datasets
    #         _window_datasets = []
    #         for dataset in datasets[1:]:
    #             dset_window = dataset.windows(wname)
    #             dset_window = transformer(dset_window)
    #             _window_datasets.append(dset_window)
    #         window_datasets.append(_window_datasets)

    #     # Merge dataset windows
    #     datasets = [
    #         multimodal_multi_merge([dataset[i] for dataset in window_datasets])
    #         for i in range(len(window_datasets[0]))
    #     ]
    #     return datasets
            

    #     raise NotImplementedError(f"Reduce_on: {reduce_on} not implemented yet")
    # else:
    #     raise ValueError(
    #         "Invalid reduce_on value. Must be one of: 'all', 'axis', 'sensor"
    #     )

def simplify_wname(wname):
    basic_tags = ['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']
    for basic_tag in basic_tags:
        if basic_tag in wname:
            return basic_tag
    return wname
