# Python imports
from typing import Any, Dict, List

# Third-party imports
from basic.config import *

# Filter warnings from UMAP
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# Librep imports
from librep.datasets.multimodal import (
    MultiModalDataset,
    TransformMultiModalDataset,
    WindowedTransform,
)

# Scaling transform
def do_scaling(
    datasets: Dict[str, MultiModalDataset],
    scaler_config: ScalerConfig,
    scale_on: str = "self",
    suffix: str = "scaled.",
    apply_only_in: List[str] = ("train_dataset", "validation_dataset", "test_dataset"),
    train_dataset_name="train_dataset",
) -> Dict[str, MultiModalDataset]:
    """Utilitary function to perform scaling to a list of datasets.
    If scale_on is "self", the scaling will be fit and transformed applied
    to each dataset. If scale_on is "train", the scaling will be fit to the
    first dataset and then, the scaling will be applied to all the datasets.
    (including the first one, that is used to fit the model).

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        The list of datasets to scale. The first dataset will be used to fit
        the scaler if scale_on is "train".
    scaler_config : ScalerConfig
        The scaler configuration, used to instantiate the scaler.
    scale_on : str, optional
        How scaler will perform, by default "self".
        It can have the following values:
        - "self": the scaler will be fit and transformed applied to each dataset.
        - "train": the scaler will be fit to the first dataset and then, the
            scaling will be applied to all the datasets.
    suffix : str, optional
        The new suffix to be appended to the window name, by default "scaled."
    apply_only_in: List[str], optional
        List of datasets to apply the scaler.
    train_dataset_name: str, optional
        If scale_on is "train", this parameter is used to specify the name of
        the dataset to use to fit the scaler.

    Returns
    -------
    Dict[str, MultiModalDataset]
        Dictonary with dataset name and the respective scaled dataset.

    Raises
    ------
    ValueError
        - If the scale_on value is invalid.
    """
    #
    kwargs = scaler_config.kwargs or {}
    if scale_on == "self":
        # Loop over the datasets
        for dataset_name in apply_only_in:
            # Get the dataset
            dataset = datasets[dataset_name]
            # Get the scaler class and instantiate it using the kwargs
            transform = scaler_cls[scaler_config.algorithm](**kwargs)
            # Fit the scaler usinf the whole dataset and (i.e., fit_on="all")
            # and then, apply the transform to the whole dataset (i.e.,
            # transform_on="all")
            windowed_transform = WindowedTransform(
                transform=transform,
                fit_on="all",
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            transformer = TransformMultiModalDataset(
                transforms=[windowed_transform], new_window_name_prefix=suffix
            )
            # Apply the transform to the dataset
            dataset = transformer(dataset)
            # Append the dataset to the list of new datasets
            datasets[dataset_name] = dataset

        return datasets

    elif scale_on == "train":
        # Get the scaler class and instantiate it using the kwargs
        transform = scaler_cls[scaler_config.algorithm](**kwargs)
        # Fit the scaler on the first dataset
        transform.fit(datasets[train_dataset_name][:][0])
        for dataset_name in apply_only_in:
            dataset = datasets[dataset_name]
            # Instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            windowed_transform = WindowedTransform(
                transform=transform,
                fit_on=None,
                transform_on="all",
            )
            # Instantiate the TransformMultiModalDataset with the list of transforms
            transformer = TransformMultiModalDataset(
                transforms=[windowed_transform], new_window_name_prefix=suffix
            )
            # Apply the transform to the dataset
            dataset = transformer(dataset)
            # Append the dataset to the list of new datasets
            datasets[dataset_name] = dataset
        return datasets
    else:
        raise ValueError(f"scale_on: {scale_on} is not valid")