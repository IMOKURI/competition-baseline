import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def make_fold(c, df):
    # num_bins = int(np.floor(1 + np.log2(len(df))))
    # df.loc[:, "bins"] = pd.cut(df["Pawpularity"], bins=num_bins, labels=False)

    fold_ = StratifiedKFold(
        n_splits=c.params.n_fold, shuffle=True, random_state=c.params.seed
    )
    for n, (train_index, val_index) in enumerate(fold_.split(df, df["label"])):
        df.loc[val_index, "fold"] = int(n)
    df["fold"] = df["fold"].astype(np.int8)
    # print(df.groupby(["fold", "bins"]).size())

    return df
