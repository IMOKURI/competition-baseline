import os

import pandas as pd

from .make_fold import make_fold


class InputData:
    def __init__(self, c):
        self.c = c
        for file_name in c.settings.inputs:
            file_path = os.path.join(c.settings.dirs.input, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                setattr(self, os.path.splitext(file_name)[0], df)

        if c.settings.debug:
            self.sample_for_debug()

        if getattr(self, "train", None) is not None:
            self.train = make_fold(c, self.train)

    def sample_for_debug(self):
        for file_name in self.c.settings.inputs:
            stem = os.path.splitext(file_name)[0]
            try:
                df = getattr(self, stem)
            except AttributeError:
                continue

            if len(df) > self.c.settings.n_debug_data:
                df = df.sample(
                    n=self.c.settings.n_debug_data, random_state=self.c.params.seed
                ).reset_index(drop=True)
                setattr(self, stem, df)
