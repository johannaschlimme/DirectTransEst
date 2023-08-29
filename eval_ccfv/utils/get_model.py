
from monai.networks.nets import UNet

def get_model(args, configs):
    if configs['model']['name'] == 'UNet':
        model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        )
    return model