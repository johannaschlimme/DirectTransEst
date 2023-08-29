import sys
import time

import torch
import wandb
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference


def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler, epoch, update_lr_scheduler=True):
    """
    Training process for one epoch.

    """

    model.train()
    step = 0
    epoch_len = len(train_loader)
    epoch_loss = 0

    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(batch_data["image"])
            loss = criterion(outputs, batch_data["label"])
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}"
        )
        wandb.log({"train/loss": loss.item()})
        sys.stdout.flush()
    
    epoch_loss /= step
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

    if update_lr_scheduler:
        lr_scheduler.step()
        wandb.log({"train/lr": lr_scheduler.get_last_lr()[0]})

    torch.cuda.empty_cache()


def evaluate(model, val_loader, dice_metric, post_trans, epoch):
    """
    Evaluation process for one epoch.

    """

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    inputs=val_data["image"],
                    roi_size=(96, 96, 96),
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.6,
                )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_data["label"])

        metric = dice_metric.aggregate().item()

        wandb.log({"val/dice_metric": metric, "epoch": epoch})
        dice_metric.reset()

    torch.cuda.empty_cache()

    return metric