
import json
import torch
import random
import numpy as np

from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged


def get_loader(args):
    data_list = json.load(open(args.data_list_file))
    val_list = data_list['val']
    val_transforms = [
        LoadImaged(keys=["data", "seg"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["data", "seg"]),
    ]
    val_transforms = Compose(val_transforms)
    val_ds = Dataset(data=val_list,
                     transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=4,
                            shuffle=False,
                            drop_last=False)

    return val_loader


def get_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        )
    return model


def load_pretrained_model(ckp_path, model):
    try:
        model.load_state_dict(torch.load(ckp_path))
        print(f'loading checkpoint from {ckp_path}')
    except:
        print("Checkpoint does not exist!")
    
    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True