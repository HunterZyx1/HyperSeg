import os
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
    best_test_acc=None,
    best_test_F1_10=None,
    best_test_F1_50=None,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "best_test_acc": best_test_acc,
        "best_test_F1_10": best_test_F1_10,
        "best_test_F1_50": best_test_F1_50,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))


def _get_map_location(device):
    if device is None:
        return "cpu"
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resume(
    result_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device=None,
) -> Tuple[Any]:

    resume_path = os.path.join(result_path, "checkpoint.pth")
    print("loading checkpoint {}".format(resume_path))

    map_location = _get_map_location(device)
    checkpoint = torch.load(resume_path, map_location=map_location)

    last_epoch = checkpoint["epoch"]
    begin_epoch = last_epoch + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    model.load_state_dict(checkpoint["state_dict"])

    # confirm whether the optimizer matches that of checkpoints
    optimizer.load_state_dict(checkpoint["optimizer"])

    if device is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(map_location)

    best_test_acc = checkpoint.get("best_test_acc", None)
    best_test_F1_10 = checkpoint.get("best_test_F1_10", None)
    best_test_F1_50 = checkpoint.get("best_test_F1_50", None)

    return (
        begin_epoch,
        model,
        optimizer,
        best_loss,
        best_test_acc,
        best_test_F1_10,
        best_test_F1_50,
    )
