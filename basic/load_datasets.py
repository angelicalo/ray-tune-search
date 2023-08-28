# Python imports
from typing import Any, Dict, List

from basic.config import *

# Librep imports
from librep.config.type_definitions import PathLike
from librep.datasets.har.loaders import PandasMultiModalLoader
from librep.datasets.multimodal import ArrayMultiModalDataset


def load_datasets(
    dataset_locations: Dict[str, PathLike],
    datasets_to_load: List[str],
    label_columns: str = "standard activity code",
    features: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
) -> ArrayMultiModalDataset:
    """Utilitary function to load the datasets.
    It load the datasets from specified in the `datasets_to_load` parameter.
    The datasets specified are concatenated into a single ArrayMultiModalDataset.
    This dataset is then returned.

    Parameters
    ----------
    datasets_to_load : List[str]
        A list of datasets to load. Each dataset is specified as a string in the
        following format: "dataset_name.dataset_view[split]". The dataset name is the name
        of the dataset as specified in the `datasets` variable in the config.py
        file. The split is the split of the dataset to load. It can be either
        "train", "validation" or "test".
    label_columns : str, optional
        The name of column that have the label, by default "standard activity code"
    features : List[str], optional
        The features to load, from datasets
        by default ( "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z", )

    Returns
    -------
    ArrayMultiModalDataset
        An ArrayMultiModalDataset with the loaded datasets (concatenated).

    Examples
    --------
    >>> load_datasets(
    ...     dataset_locations={
    ...         "kuhar.standartized_balanced": "data/kuhar",
    ...         "motionsense.standartized_balanced": "data/motionsense",
    ...     },
    ...     datasets_to_load=[
    ...         "kuhar.standartized_balanced[train]",
    ...         "kuhar.standartized_balanced[validation]",
    ...         "motionsense.standartized_balanced[train]",
    ...         "motionsense.standartized_balanced[validation]",
    ...     ],
    ... )
    """
    # Transform it to a Path object
    dset_names = set()

    # Remove the split from the dataset name
    # dset_names will contain the name of the datasets to load
    # it is used to index the datasets variable in the config.py file
    for dset in datasets_to_load:
        name = dset.split("[")[0]
        dset_names.add(name)

    # Load the datasets
    multimodal_datasets = dict()
    for name in dset_names:
        # Define dataset path. Join the root_dir with the path of the dataset
        path = dataset_locations[name]
        # Load the dataset
        loader = PandasMultiModalLoader(root_dir=path)
        train, validation, test = loader.load(
            load_train=True,
            load_validation=True,
            load_test=True,
            as_multimodal=True,
            as_array=True,
            features=features,
            label=label_columns,
        )
        # Store the multiple MultimodalDataset in a dictionary
        multimodal_datasets[name] = {
            "train": train,
            "validation": validation,
            "test": test,
        }

    # Concatenate the datasets

    # Pick the name and the split of the first dataset to load
    name = datasets_to_load[0].split("[")[0]
    split = datasets_to_load[0].split("[")[1].split("]")[0]
    final_dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])

    # Pick the name and the split of the other datasets to load and
    # Concatenate the other datasets
    for dset in datasets_to_load[1:]:
        name = dset.split("[")[0]
        split = dset.split("[")[1].split("]")[0]
        dset = ArrayMultiModalDataset.from_pandas(multimodal_datasets[name][split])
        final_dset = ArrayMultiModalDataset.concatenate(final_dset, dset)

    return final_dset