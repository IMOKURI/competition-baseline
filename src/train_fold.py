import logging
import time

import mlflow
import numpy as np
import torch.cuda.amp as amp
import wandb

from .get_score import get_score
from .make_dataset import make_dataloader, make_dataset
from .make_loss import make_criterion, make_optimizer, make_scheduler
from .make_model import make_model
from .train_epoch import train_epoch, validate_epoch
from .utils import EarlyStopping

log = logging.getLogger("__main__").getChild("train_loop")


def train_fold(c, df, fold, device):
    # ====================================================
    # Data Loader
    # ====================================================
    trn_idx = df[df["fold"] != fold].index
    val_idx = df[df["fold"] == fold].index

    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx].reset_index(drop=True)

    train_ds = make_dataset(c, train_folds, "train")
    valid_ds = make_dataset(c, valid_folds, "valid")

    train_loader = make_dataloader(c, train_ds, shuffle=True, drop_last=True)
    valid_loader = make_dataloader(c, valid_ds, shuffle=False, drop_last=False)

    # ====================================================
    # Model
    # ====================================================
    model = make_model(c, device)

    criterion = make_criterion(c)
    optimizer = make_optimizer(c, model)
    scaler = amp.GradScaler(enabled=c.settings.amp)
    scheduler = make_scheduler(c, optimizer, train_ds)

    es = EarlyStopping(
        patience=c.params.es_patience,
        verbose=True,
        path=f"{c.params.model_name.replace('/', '-')}_fold{fold}",
        mlflow=c.mlflow.enabled,
    )

    # ====================================================
    # Loop
    # ====================================================
    for epoch in range(c.params.epoch):
        start_time = time.time()

        # train
        avg_loss = train_epoch(
            c,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            device,
        )

        # eval
        avg_val_loss, preds = validate_epoch(c, valid_loader, model, criterion, device)
        valid_labels = valid_folds["label"].values

        if "WithLogitsLoss" in c.params.criterion:
            preds = 1 / (1 + np.exp(-preds))

        # scoring
        if c.settings.n_class == 1:
            score = get_score(c.settings.scoring, valid_labels, preds)
        elif c.settings.n_class > 1:
            score = get_score(c.settings.scoring, valid_labels, preds.argmax(1))
        else:
            raise Exception("Invalid n_class.")

        elapsed = time.time() - start_time
        log.info(
            f"Epoch {epoch+1} - "
            f"loss_train: {avg_loss:.4f} "
            f"loss_val: {avg_val_loss:.4f} "
            f"score: {score:.4f} "
            f"time: {elapsed:.0f}s"
        )
        if c.mlflow.enabled:
            mlflow.log_metrics(
                {
                    f"loss_train_{fold}": avg_loss,
                    f"loss_val_{fold}": avg_val_loss,
                    f"score_{fold}": score,
                },
                step=epoch,
            )
        if c.wandb.enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    f"loss/train_fold{fold}": avg_loss,
                    f"loss/valid_fold{fold}": avg_val_loss,
                    f"score/fold{fold}": score,
                }
            )

        es(avg_val_loss, score, model, preds)

        if es.early_stop:
            log.info("Early stopping")
            break

    if c.settings.n_class == 1:
        valid_folds["preds"] = es.best_preds
    elif c.settings.n_class > 1:
        valid_folds[[str(c) for c in range(c.settings.n_class)]] = es.best_preds
        valid_folds["preds"] = es.best_preds.argmax(1)
    else:
        raise Exception("Invalid n_class.")

    return valid_folds, es.best_score, es.best_loss
