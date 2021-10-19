import os

import pandas as pd


def load_data(c):
    train = pd.read_csv(os.path.join(c.settings.dirs.input, "train.csv"))
    # test = pd.read_csv(os.path.join(c.settings.dirs.input, "test.csv"))
    # sub = pd.read_csv(os.path.join(c.settings.dirs.input, "sample_submission.csv"))

    if c.settings.debug:
        if len(train) > c.settings.n_debug_data:
            train = train.sample(
                n=c.settings.n_debug_data, random_state=c.params.seed
            ).reset_index(drop=True)
        # if len(test) > c.settings.n_debug_data:
        #     test = test.sample(
        #         n=c.settings.n_debug_data, random_state=c.params.seed
        #     ).reset_index(drop=True)
        # if len(sub) > c.settings.n_debug_data:
        #     sub = sub.sample(
        #         n=c.settings.n_debug_data, random_state=c.params.seed
        #     ).reset_index(drop=True)

    # return train, test, sub
    return train, None, None
