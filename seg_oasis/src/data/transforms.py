from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityd,
    Spacingd,
    MapTransform,
    ToDeviced,
)
import torch
import numpy as np


class ConvertSynthSegClassesToSingleLabel(MapTransform):
    """
    Convert given SynthSeg classes to single label. For SynthSeg classes, see:
    https://github.com/BBillot/SynthSeg/blob/master/data/labels%20table.txt
    
    """

    def __init__(self, labels_of_interest):
        self.labels_of_interest = labels_of_interest

    def __call__(self, data):
        modified_labels = torch.zeros_like(data["label"])

        for i in range(len(modified_labels)):
            current_label = data["label"][i]
            modified_labels[i][np.isin(current_label, self.labels_of_interest)] = 1

        data["label"] = modified_labels
        return data


def get_train_transforms(config, device):
    """
    Transforms for training data.
    
    """

    if config["segmentation_type"] == "white_matter":
        labels_of_interest = [2, 41]
    elif config["segmentation_type"] == "hippocampus":
        labels_of_interest = [17, 53]
    elif config["segmentation_type"] == "ventricular_system":
        labels_of_interest = [4, 5, 14, 15, 43, 44]


    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToDeviced(keys=["image", "label"], device=device),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ConvertSynthSegClassesToSingleLabel(labels_of_interest),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=(128, 128, 128), random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            ScaleIntensityd(keys="image", channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

    return train_transforms


def get_val_transforms(config, device):
    """
    Transforms for validation data.

    """

    if config["segmentation_type"] == "white_matter":
        labels_of_interest = [2, 41]
    elif config["segmentation_type"] == "hippocampus":
        labels_of_interest = [17, 53]
    elif config["segmentation_type"] == "ventricular_system":
        labels_of_interest = [4, 5, 14, 15, 43, 44]
    

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToDeviced(keys=["image", "label"], device=device),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ConvertSynthSegClassesToSingleLabel(labels_of_interest),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityd(keys="image", channel_wise=True),
        ]
    )

    return val_transforms
