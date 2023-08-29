import argparse
import os
import sys
import time

import torch
import wandb
from utils.checkpoints import save_checkpoint
from utils.train_func import evaluate, train
from data.dataloaders import get_train_tl_loader, get_val_tl_loader
from dotenv import load_dotenv
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism


def main_worker(args, config):
    """
    Main worker for training. Setup training environment, initialize model, loss function, optimizer, scheduler, and execute PyTorch training process.

    """

    # Set up device and GradScaler for mixed precision training
    device = torch.device(f"cuda:0")
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    total_start = time.time()

    # Initialize WandB experiment
    wandb.init(project="medseg-transfer-learning", group="adni", name=args.run_name, config=config)

    # Load pre-trained model from artifact directory
    artifact_dir = config["artifact_dir"]
    
    model_path = os.path.join(artifact_dir, args.artifact_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact '{args.artifact_name}' not found in {artifact_dir}")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path))

    # Freezing first two blocks of the encoder
    for name, param in model.named_parameters():
        if "model.0" in name or "model.1.submodule.0" in name:
            param.requires_grad = False

    # Define loss function, optimizer, and learning rate scheduler
    loss_function = DiceLoss(
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True,
        batch=True,
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Obtain data loaders for training and validation
    train_loader = get_train_tl_loader(config, device)
    val_loader = get_val_tl_loader(config, device)

    # Log gradients to WandB
    wandb.watch(model, log_freq=25)

    # Set up checkpoint directory
    checkpoint_dir = os.path.join(config["checkpoint_dir"], args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_dice_metric = -1
    best_dice_metric_epoch = -1

    # Start training process
    print(f"time elapsed before training: {time.time() - total_start}")
    train_start = time.time()
    for epoch in range(config["max_epochs"]):
        if epoch % args.n_epoch_checks == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, checkpoint_dir)

        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config['max_epochs']}")

        # Train the model
        train(train_loader, model, loss_function, optimizer, lr_scheduler, scaler, epoch, update_lr_scheduler=False)

        # Evaluate on validation set
        if (epoch + 1) % config["val_interval"] == 0:
            metric = evaluate(
                model, val_loader, dice_metric, post_trans, epoch
            )
            lr_scheduler.step(metric)

            # Save the best model based on validation metric
            if metric > best_dice_metric:
                best_dice_metric = metric
                best_dice_metric_epoch = epoch + 1
                model_filename = f"{args.run_name}.pth"
                model_path = os.path.join(config["model_dir"], model_filename)
                torch.save(model.state_dict(), model_path)

        print(
            f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}"
        )
        sys.stdout.flush()

    print(f"train completed, total train time: {(time.time() - train_start):.4f}")
    wandb.log({"best_dice_metric": best_dice_metric,
                "best_dice_metric_epoch": best_dice_metric_epoch})

    # Version the best model and save as artifact to WandB
    best_model_path = os.path.join(config["model_dir"], f"{args.run_name}.pth")
    model_artifact = wandb.Artifact(
        name=f"{args.run_name}.pth",
        type="model",
        description=f"(Transfer Learning) {config['network']} for 3D Segmentation of {config['segmentation_type']} in the brain",
    )
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()


def main():
    """
    Main function for training. Parse command-line arguments and call main_worker function.

    """

    # Load environment variables
    load_dotenv()
    data_dir = os.environ.get("DATA_DIRECTORY")
    model_dir = os.environ.get("MODEL_DIRECTORY")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIRECTORY")
    artifact_dir = os.environ.get("ARTIFACT_DIRECTORY")

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=data_dir, type=str, help="directory of dataset")
    parser.add_argument("--model_dir", default=model_dir, type=str, help="directory to save best model")
    parser.add_argument("--checkpoint_dir", default=checkpoint_dir, type=str, help="directory to save checkpoints")
    parser.add_argument("--artifact_dir", default=artifact_dir, type=str, help="directory to save model artifact from wandb")
    parser.add_argument("--epochs", default=3, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size of every GPU")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="test_run")
    parser.add_argument("--segmentation_type", type=str, default="white_matter", choices=["white_matter", "hippocampus", "ventricular_system"] , help="type of segmentation task")
    parser.add_argument("--n_epoch_checks", type=int, default=5, help="number of epochs before saving checkpoint")
    parser.add_argument("--artifact_name", type=str, default="adni_base_wm.pth", help="name of artifact")
    parser.add_argument("--current_fold", type=int, default=0, help="current fold of cross-validation")
    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        set_determinism(seed=args.seed)

    print(args)

    # Set up config dictionary
    config = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "artifact_dir": args.artifact_dir,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "val_interval": args.val_interval,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "lr_scheduler": "cosine_annealing",
        "loss_function": "DiceLoss",
        "val_amp": True,
        "seed": args.seed,
        "network": "UNet",
        "segmentation_type": args.segmentation_type,
        "artifact": args.artifact_name,
        "current_fold": args.current_fold,
    }

    main_worker(args=args, config=config)


if __name__ == "__main__":
    main()