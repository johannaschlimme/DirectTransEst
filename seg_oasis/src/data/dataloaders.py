import os
import glob
from monai.data import DataLoader, Dataset

from .transforms import get_train_transforms, get_val_transforms


def get_train_ri_loader(config, device):
    current_fold = config["current_fold"]

    for fold in range(3):
        if fold != current_fold:
            train_dir = os.path.join(config["data_dir"], f"oasis/fold_{fold}")

    train_images = sorted(glob.glob(os.path.join(train_dir, "*/*/*T1toMNIlin.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_dir, "*/*/*T1toMNIlin_synthseg.nii.gz")))
    train_dict = [{"image": img, "label": seg} for img, seg in zip(train_images, train_labels)]

    train_ds = Dataset(
        data=train_dict,
        transform=get_train_transforms(config, device),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    return train_loader


def get_train_tl_loader(config, device):
    current_fold = config["current_fold"]

    for fold in range(3):
        if fold != current_fold:
            train_dir = os.path.join(config["data_dir"], f"oasis/train_100/fold_{fold}")

    train_images = sorted(glob.glob(os.path.join(train_dir, "*/*/*T1toMNIlin.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(train_dir, "*/*/*T1toMNIlin_synthseg.nii.gz")))
    train_dict = [{"image": img, "label": seg} for img, seg in zip(train_images, train_labels)]

    train_ds = Dataset(
        data=train_dict,
        transform=get_train_transforms(config, device),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    return train_loader


def get_val_ri_loader(config, device):
    current_fold = config["current_fold"]

    val_dir = os.path.join(config["data_dir"], f"oasis/fold_{current_fold}")
    val_images = sorted(glob.glob(os.path.join(val_dir, "*/*/*T1toMNIlin.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(val_dir, "*/*/*T1toMNIlin_synthseg.nii.gz")))
    val_dict = [{"image": img, "label": seg} for img, seg in zip(val_images, val_labels)]

    val_ds = Dataset(
        data=val_dict,
        transform=get_val_transforms(config, device),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    return val_loader


def get_val_tl_loader(config, device):
    current_fold = config["current_fold"]

    val_dir = os.path.join(config["data_dir"], f"oasis/train_100/fold_{current_fold}")
    val_images = sorted(glob.glob(os.path.join(val_dir, "*/*/*T1toMNIlin.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(val_dir, "*/*/*T1toMNIlin_synthseg.nii.gz")))
    val_dict = [{"image": img, "label": seg} for img, seg in zip(val_images, val_labels)]

    val_ds = Dataset(
        data=val_dict,
        transform=get_val_transforms(config, device),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    return val_loader
