import json
import logging
import math
import os
import random
import time

import git
import numpy as np
import requests
import torch
import wandb
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

log = logging.getLogger("__main__").getChild("utils")


def fix_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def debug_settings(c):
    if c.settings.debug:
        c.wandb.enabled = False
        c.settings.print_freq = 10
        c.params.n_fold = 3
        c.params.epoch = 1


def gpu_settings(c):
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = c.settings.gpus
        log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    except ConfigAttributeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"torch device: {device}, device count: {torch.cuda.device_count()}, CUDA version: {torch.version.cuda}")
    return device


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf
        self.delta = delta
        self.path = path
        self.best_preds = None

    def __call__(self, val_loss, score, model, preds):

        if self.best_score is None:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model)
        elif val_loss >= self.best_loss + self.delta:
            if self.patience <= 0:
                return
            self.counter += 1
            log.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_preds = preds
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            log.info(
                f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), f"{self.path}_best.pth")
        self.best_loss = val_loss


def compute_grad_norm(parameters, norm_type=2.0):
    """Refer to torch.nn.utils.clip_grad_norm_"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device)
             for p in parameters]
        ),
        norm_type,
    )

    return total_norm


def setup_wandb(c):
    if c.wandb.enabled:
        os.makedirs(os.path.abspath(c.wandb.dir), exist_ok=True)
        c_dict = OmegaConf.to_container(c.params, resolve=True)
        c_dict["commit"] = get_commit_hash(c.settings.dirs.working)
        run = wandb.init(
            entity=c.wandb.entity,
            project=c.wandb.project,
            dir=os.path.abspath(c.wandb.dir),
            config=c_dict,
        )
        return run


def teardown_wandb(c, run, loss):
    if c.wandb.enabled:
        wandb.summary["loss"] = loss
        artifact = wandb.Artifact(
            c.params.model_name.replace("/", "-"), type="model")
        artifact.add_dir(".")
        run.log_artifact(artifact)


def get_commit_hash(dir_):
    repo = git.Repo(dir_, search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def send_result_to_slack(c, score, loss):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    msg = f"score: {score:.5f}, loss: {loss:.5f}, model: {c.params.model_name}"
    try:
        requests.post(webhook_url, data=json.dumps({"text": msg}))
    except Exception:
        log.warning(f"Failed to send message to slack.")
