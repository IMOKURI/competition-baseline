import logging
import time
import warnings

import numpy as np
import torch
import torch.cuda.amp as amp

from .utils import AverageMeter, compute_grad_norm, timeSince

log = logging.getLogger("__main__").getChild("train_epoch")


def train_epoch(
    c, train_loader, model, criterion, optimizer, scheduler, scaler, epoch, device
):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with amp.autocast(enabled=c.settings.amp):
            # y_preds = model(images)
            y_preds = model(images).squeeze(1)

            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)
            loss = loss / c.params.gradient_acc_step

        scaler.scale(loss).backward()

        if (step + 1) % c.params.gradient_acc_step == 0:
            scaler.unscale_(optimizer)

            # error_if_nonfinite に関する warning を抑止する
            # pytorch==1.10 で不要となりそう
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c.params.max_grad_norm, error_if_nonfinite=False
                )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            grad_norm = compute_grad_norm(model.parameters())

        # end = time.time()
        if step % c.settings.print_freq == 0 or step == (len(train_loader) - 1):
            log.info(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(train_loader)):s} "
                f"Loss: {losses.avg:.4f} "
                f"Grad: {grad_norm:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.2e}  "
                # f"LR: {scheduler.get_lr()[0]:.2e}  "
            )

    return losses.avg


def validate_epoch(c, valid_loader, model, criterion, device):
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()
    preds = []
    start = time.time()

    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # with torch.inference_mode():
        with torch.no_grad():
            # y_preds = model(images)
            y_preds = model(images).squeeze(1)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        if c.settings.n_class == 1:
            preds.append(y_preds.to("cpu").numpy())
        elif c.settings.n_class > 1:
            preds.append(y_preds.softmax(1).to("cpu").numpy())
        else:
            raise Exception("Invalid n_class.")

        # end = time.time()
        if step % c.settings.print_freq == 0 or step == (len(valid_loader) - 1):
            log.info(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed {timeSince(start, float(step + 1) / len(valid_loader)):s} "
                f"Loss: {losses.avg:.4f} "
            )

    predictions = np.concatenate(preds)
    return losses.avg, predictions
