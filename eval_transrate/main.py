# %%
import argparse
import os

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from utils.transrate import cal_transrate
from utils.utility import (get_loader, get_model, load_pretrained_model,
                           setup_seed)


# %%
def evaluate(test_loader, model, device):
    model.eval()
    print("begin evaluation!")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    all_transrates = []
    with torch.no_grad():
        for val_data in test_loader:
            features = sliding_window_inference(
                inputs=val_data["data"].to(device),
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.6,
            )
            features = [post_trans(i) for i in decollate_batch(features)]

            transrates_per_image = []
            # Split features and labels into 2D slices along z-axis for transrate calculation
            for z in range(features[0].shape[2]):
                val_output_slice = [output[:, :, z].cpu().numpy() for output in features]
                label_slice = [label[:, :, z].cpu().numpy() for label in val_data["seg"]]
            
                for feature, label in zip(val_output_slice, label_slice):
                    feature = feature.reshape(-1, 1)
                    label = label.reshape(-1, 1) 

                    transrate_value = cal_transrate(feature, label)
                    transrates_per_image.append(transrate_value)
                
            # Compute average transrate for the current 3D image and store
            avg_transrate_per_image = np.mean(transrates_per_image)
            print(f"transrate for current 3D image: {avg_transrate_per_image}")
            all_transrates.append(avg_transrate_per_image)

    final_transrate = np.mean(all_transrates)
    print(f"transrate: {final_transrate}")


# %%
def main():
    parser = argparse.ArgumentParser(description='PyTorch Evaluation')
    parser.add_argument('--data_list_file', type=str,
                        help='json file of data paths')
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    setup_seed(0)
    print(args)
    test_loader = get_loader(args)
    model = get_model()
    model.to(device)

    ckp_path = args.model_path
    # check if ckp_path exists
    if ckp_path != '' and os.path.exists(ckp_path):
        model = load_pretrained_model(ckp_path, model)
        evaluate(test_loader, model, device)
    else:
        print("Please provide the correct path to your checkpoint!")


# %%
if __name__ == "__main__":
    main()