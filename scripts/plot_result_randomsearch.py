from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from matplotlib import cm
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import load_experiment, load_experiments_df

from tabrepo.loaders import Paths
import os
print(os.getcwd())


# if __name__ == '__main__':
expnames = {
    "cMIor": 2,
    "gpog0": 5,
}
n = 1000
for expname, n_splits in expnames.items():
    name_filter = lambda path: expname in str(path)
    df_random = load_experiments_df(path_filter=name_filter)

    for fold in df_random.fold.unique():
        df_small = df_random[df_random.fold == fold]
        df_small = df_small.sample(frac=1)
        errors = df_small[['train_error', 'test_error']]
        fig = seaborn.regplot(errors.head(n), x='train_error', y='test_error').figure
        fig.suptitle(f"Correlation plots between train and test error ({n_splits} splits)")
        plt.tight_layout()
        plt.show()

        errors = df_small[['train_error', 'test_error']]
        corr = errors.head(n).corr()
        print(f"Pearson correlation between train and test error ({n_splits} splits): {corr.min().min()}")

