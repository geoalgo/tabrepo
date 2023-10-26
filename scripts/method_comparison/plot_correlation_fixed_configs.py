"""
Compare different methods that searches for ensemble configurations given offline evaluations.
Several strategies are available:
* all: evaluate the ensemble performance when using all model available
* zeroshot: evaluate the ensemble performance of zeroshot configurations
* zeroshot-ensemble: evaluate the ensemble performance of zeroshot configurations and when scoring list of models with
their ensemble performance
* randomsearch: performs a randomsearch after initializing the initial configuration with zeroshot
* localsearch: performs a localsearch after initializing the initial configuration with zeroshot. For each new
candidate, the best current configuration is mutated.

For random/local search, the search is done asynchronously with multiple workers.

Example:
PYTHONPATH=. python scripts/run_method_comparison.py --setting slow --n_workers 64
"""
import ast
import logging
import os
import shutil
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from tabrepo.contexts.context_2022_10_13 import load_context_2022_10_13
from tabrepo.contexts.context_2022_12_11_bag import load_context_2022_12_11_bag
from tabrepo.loaders import Paths
from tabrepo.simulation.filter_dataset_correlation import sort_datasets_linkage
from tabrepo.utils import catchtime
from scripts.method_comparison.evaluate_ensemble import evaluate_ensemble

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


def load_configs(fold: int, expname: str) -> List[str]:
    csv_filename = Paths.results_root / f"{expname}.csv"
    results_df = pd.read_csv(csv_filename)
    results_df = results_df[results_df.fold == fold]
    configs = [ast.literal_eval(x) for x in results_df.selected_configs.values]
    return configs


if __name__ == "__main__":
    with catchtime("load"):
        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_2022_12_11_bag(load_zeroshot_pred_proba=True, lazy_format=True)
    expname = "pYAMa"
    ensemble_size = 10
    datasets = zsc.get_datasets()
    all_datasets = np.array(datasets)
    np.random.shuffle(all_datasets)
    n_splits = 5
    kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    splits = kf.split(all_datasets)
    (Paths.results_root / f"clustermap.csv").unlink(missing_ok=True)
    # Evaluate all search strategies on `n_splits` of the datasets. Results are logged in a csv and can be
    # analysed with plot_results_comparison.py.
    results = []
    for i, (train_index, test_index) in tqdm(enumerate(splits), total=n_splits):
        list_configs = load_configs(expname=expname, fold=i + 1)
        for configs in list_configs:
            with catchtime(f'Eval config'):
                train_datasets = list(all_datasets[train_index])
                cutoff = len(train_datasets) * 7 // 10
                sub_train_datasets = sort_datasets_linkage(zsc, train_datasets)[:cutoff]
                test_datasets = list(all_datasets[test_index])

                for use_subset in [False, True]:
                    train_datasets_selected = sub_train_datasets if use_subset else train_datasets
                    train_error, test_error = evaluate_ensemble(
                        configs=configs,
                        train_datasets=zsc.get_dataset_folds(train_datasets_selected),
                        test_datasets=zsc.get_dataset_folds(test_datasets),
                        num_folds=10,
                        ensemble_size=ensemble_size,
                        backend="ray",
                    )
                    print(f"train/test error of config found: {train_error}/{test_error}")
                    results.append({
                        'fold': i + 1,
                        'train-score': train_error,
                        'test-score': test_error,
                        'config': configs,
                        'subselect': use_subset
                    })
                    print(results[-1])
                    csv_filename = Paths.results_root / f"clustermap-{expname}.csv"
                    print(f"update results in {csv_filename}")
                    pd.DataFrame(results).to_csv(csv_filename, index=False)
    print(results)
