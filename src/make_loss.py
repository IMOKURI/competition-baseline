import torch.nn as nn
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts)


# ====================================================
# Criterion
# ====================================================
def make_criterion(c):
    if c.params.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif c.params.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif c.params.criterion == "MSELoss":
        criterion = nn.MSELoss()

    else:
        raise Exception("Invalid criterion.")
    return criterion


# ====================================================
# Optimizer
# ====================================================
def make_optimizer(c, model):
    if c.params.optimizer == "Adam":
        optimizer = Adam(
            model.parameters(), lr=c.params.lr, weight_decay=c.params.weight_decay
        )
    elif c.params.optimizer == "AdamW":
        optimizer = AdamW(
            model.parameters(), lr=c.params.lr, weight_decay=c.params.weight_decay
        )

    else:
        raise Exception("Invalid optimizer.")
    return optimizer


# ====================================================
# Scheduler
# ====================================================
def make_scheduler(c, optimizer, ds):
    num_data = len(ds)
    num_steps = (
        num_data // (c.params.batch_size * c.params.gradient_acc_step) * c.params.epoch
    )

    if c.params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=num_steps, T_mult=1, eta_min=c.params.min_lr, last_epoch=-1
        )
    elif c.params.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=c.params.min_lr, last_epoch=-1
        )
    elif c.params.scheduler == "CosineAnnealingWarmupRestarts":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_steps,
            max_lr=c.params.lr,
            min_lr=c.params.min_lr,
            warmup_steps=(num_steps // 10),
        )

    else:
        raise Exception("Invalid scheduler.")
    return scheduler
