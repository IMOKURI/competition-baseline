import logging

import numpy as np
import wandb
from sklearn.metrics import accuracy_score, mean_squared_error

log = logging.getLogger("__main__").getChild("get_score")


def get_score(scoring, y_true, y_pred):
    if scoring == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif scoring == "accuracy":
        return accuracy_score(y_true, y_pred)

    else:
        raise Exception("Invalid scoring.")


def get_result(c, df, fold, loss=None):
    if c.settings.scoring == "jaccard":
        score = df["jaccard"].mean()
    else:
        preds = df["preds"].values
        labels = df["label"].values
        score = get_score(c.settings.scoring, labels, preds)

    log.info(f"Score: {score:<.5f}")
    if c.wandb.enabled:
        wandb.log({"score": score, "fold": fold})
        if loss is not None:
            wandb.log({"loss": loss, "fold": fold})

    return score
