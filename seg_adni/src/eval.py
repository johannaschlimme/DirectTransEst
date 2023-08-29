import argparse
import os
import sys
import time

import torch
import wandb
from data.dataloaders import get_val_tl_loader
from utils.train_func import evaluate
from dotenv import load_dotenv
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism


def main_worker(args, config):
    """
    Main worker for evaluation.

    """

    device = torch.device(f"cuda:0")

    total_start = time.time()

    # Initialize WandB experiment
    wandb.init(project="medseg-transfer-learning", group="adni", name=args.run_name, config=config)

    # Download and load pre-trained model from WandB artifact
    artifact = wandb.use_artifact(config["artifact"], type='model')
    artifact_dir = artifact.download(root=config["artifact_dir"])

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(artifact_dir, args.artifact_name)))

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Obtain data loader for validation
    val_loader = get_val_tl_loader(config, device)

    # Start evaluation process
    print(f"time elapsed before evaluation: {time.time() - total_start}")
    train_start = time.time()

    # Evaluate on validation set
    evaluate(model, val_loader, dice_metric, post_trans, epoch=0)

    print(f"eval completed, total time: {(time.time() - train_start):.4f}")
    sys.stdout.flush()

    wandb.finish()


def main():
    """
    Main function for evaluation. Parse command-line arguments and call main_worker function.

    """

    # Load environment variables
    load_dotenv()
    data_dir = os.environ.get("DATA_DIRECTORY")
    model_dir = os.environ.get("MODEL_DIRECTORY")
    artifact_dir = os.environ.get("ARTIFACT_DIRECTORY")

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=data_dir, type=str, help="directory of dataset")
    parser.add_argument("--model_dir", default=model_dir, type=str, help="directory to save best model")
    parser.add_argument("--artifact_dir", default=artifact_dir, type=str, help="directory to save model artifact from wandb")
    parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size of every GPU")
    parser.add_argument("--run_name", type=str, default="test_run")
    parser.add_argument("--segmentation_type", type=str, default="white_matter", choices=["white_matter", "hippocampus", "ventricular_system"] , help="type of segmentation task")
    parser.add_argument("--artifact", type=str, default="johannaschlimme/medseg-transfer-learning/UNet:v0", help="name of artifact path in wandb")
    parser.add_argument("--artifact_name", type=str, default="best_metric_model.pth", help="name of artifact")
    parser.add_argument("--current_fold", type=int, default=0, help="current fold of cross-validation")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")    
    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        set_determinism(seed=args.seed)

    print(args)

    # Set up config dictionary
    config = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "artifact_dir": args.artifact_dir,
        "batch_size": args.batch_size,
        "segmentation_type": args.segmentation_type,
        "artifact": args.artifact,
        "current_fold": args.current_fold,
        "seed": args.seed,
    }

    main_worker(args=args, config=config)


if __name__ == "__main__":
    main()