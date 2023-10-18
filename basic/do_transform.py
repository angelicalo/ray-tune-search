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

# Non-parametric transform
def do_transform(
    datasets: List[MultiModalDataset],
    transform_configs: List[TransformConfig],
    keep_suffixes: bool = True,
) -> Dict[str, MultiModalDataset]:
    """Utilitary function to apply a list of transforms to a list of datasets

    Parameters
    ----------
    datasets : List[MultiModalDataset]
        List of the datasets to transform.
    transform_configs : List[TransformConfig]
        List of the transforms to apply. Each transform it will be instantiated
        based on the TransformConfig object and each one will be applied to the
        datasets.
    keep_suffixes : bool, optional
        Keep the window name suffixes, by default True

    Returns
    -------
    List[MultiModalDataset]
        The transformed datasets.
    """
    new_datasets = dict()
    # Loop over the datasets
    for dset_name, dset in datasets.items():
        transforms = []
        new_names = []

        # Loop over the transforms and instantiate them
        for transform_config in transform_configs:
            # Get the transform class and kwargs and instantiate the transform
            kwargs = transform_config.kwargs or {}
            the_transform = transforms_cls[transform_config.transform](**kwargs)
            # If the transform is windowed, instantiate the WindowedTransform
            # with the defined fit_on and transform_on.
            if transform_config.windowed:
                the_transform = WindowedTransform(
                    transform=the_transform,
                    fit_on=transform_config.windowed.fit_on,
                    transform_on=transform_config.windowed.transform_on,
                )
            # Else instantiate the WindowedTransform with fit_on=None and
            # transform_on="all", i.e. the transform will be applied to
            # whole dataset.
            else:
                the_transform = WindowedTransform(
                    transform=the_transform,
                    fit_on=None,
                    transform_on="window",
                )
            # Create the list of transforms to apply to the dataset
            transforms.append(the_transform)
            if keep_suffixes:
                new_names.append(transform_config.name)
        
        new_name_prefix = ".".join(new_names)
        if new_name_prefix:
            new_name_prefix += "."

        # Instantiate the TransformMultiModalDataset with the list of transforms
        transformer = TransformMultiModalDataset(
            transforms=transforms, new_window_name_prefix=new_name_prefix
        )
        # Apply the transforms to the dataset
        dset = transformer(dset)
        # Append the transformed dataset to the list of new datasets
        new_datasets[dset_name] = dset

    # Return the list of transformed datasets
    return new_datasets