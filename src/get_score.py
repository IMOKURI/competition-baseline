import logging

import mlflow
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, mean_squared_error

log = logging.getLogger("__main__").getChild("get_score")


def get_score(y_true, y_pred):
    # RMSE
    # return np.sqrt(mean_squared_error(y_true, y_pred))

    # Accuracy
    return accuracy_score(y_true, y_pred)


def get_result(c, df, fold):
    preds = df["preds"].values
    labels = df["label"].values
    score = get_score(labels, preds)
    log.info(f"Score: {score:<.5f}")
    if c.mlflow.enabled:
        mlflow.log_metric("score", score, fold)
    if c.wandb.enabled:
        wandb.log({"score": score, "fold": fold})
    return score
