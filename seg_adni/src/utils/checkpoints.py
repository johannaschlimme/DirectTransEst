import os
import torch


def save_checkpoint(model, optimizer, lr_scheduler, epoch, checkpoint_dir):
    """
    Save checkpoint from current epoch, model, optimizer, and lr_scheduler under given checkpoint directory.
    
    """

    checkpoint_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(checkpoint_dict, os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
    """
    Load checkpoint from given checkpoint path and load it to model, optimizer, and lr_scheduler to be able to resume training from that checkpoint.
    Returns [model, optimizer, scheduler, epoch].
    
    """

    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint["epoch"]

    state_dicts = {
        "model": {k.replace("module.", ""): v for k, v in checkpoint["model"].items()},
        "optimizer": {
            k.replace("module.", ""): v for k, v in checkpoint["optimizer"].items()
        },
        "lr_scheduler": {
            k.replace("module.", ""): v for k, v in checkpoint["lr_scheduler"].items()
        },
    }

    model.load_state_dict(state_dicts["model"])
    optimizer.load_state_dict(state_dicts["optimizer"])
    lr_scheduler.load_state_dict(state_dicts["lr_scheduler"])

    return model, optimizer, lr_scheduler, epoch

