from __future__ import annotations

import pandas as pd
import numpy as np

from typing import List, Tuple

from autogluon.core.metrics import get_metric, Scorer
from autogluon_zeroshot.repository import EvaluationRepository
from autogluon.tabular import TabularPredictor


def obtain_input_data_for_l2(repo: EvaluationRepository, l1_models: List[str], dataset: str, fold: int) \
        -> Tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, Scorer, List[str]]:
    """
    Obtain the input data for the next stacking layer from the repository.
    Additionally, return the eval metric for this fold and dataset combination.

    :param repo: EvaluationRepository
    :param l1_models: List of models in layer 1 to use for leakage analysis.
    :param dataset: Dataset to use for leakage analysis.
    :param fold: outer fold used in this analysis
    :return: l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names
    """

    # Simulation Setup
    tid = repo.dataset_to_tid(dataset)
    task = repo.task_name(tid=tid, fold=fold)
    zsc = repo._zeroshot_context
    tid = zsc.dataset_name_to_tid_dict[task]
    task_ground_truth_metadata: dict = repo._ground_truth[tid][fold]

    problem_type = task_ground_truth_metadata['problem_type']
    metric_name = task_ground_truth_metadata['eval_metric']
    eval_metric = get_metric(metric=metric_name, problem_type=problem_type)

    # - Obtain X and y
    # - Obtain X
    train_data, test_data = repo.get_data(tid, fold)
    l2_X_train, l2_y_train, l2_X_test, l2_y_test = repo.preprocess_data(tid, fold, train_data, test_data)

    # Previous code had `y_test.fillna(-1)` in code. Not sure why, hence see where this happens with the assert.
    assert l2_y_test.hasnans is False
    l2_y_train = l2_y_train.to_numpy()
    l2_y_test = l2_y_test.fillna(-1).to_numpy()

    # - Obtain preds and build stack_X_train, stack_X_test
    pred_val, pred_test = repo._tabular_predictions.predict(dataset=tid, fold=fold, splits=['val', 'test'],
                                                            models=l1_models, force_1d=problem_type == 'binary')
    oof_col_names = []
    for i, m in enumerate(l1_models):

        if problem_type != 'binary':
            raise NotImplementedError  # TODO support multiclass here

        pred_val_m = pred_val[i]
        pred_test_m = pred_test[i]
        col_name = f'L1/OOF/{m}'

        l2_X_train[col_name] = pred_val_m
        l2_X_test[col_name] = pred_test_m
        oof_col_names.append(col_name)

    return l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names


def autogluon_l2_runner(l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric: Scorer, oof_col_names: List[str],
                        sample=20000, sample_test=20000):
    print("Start running AutoGluon on the L2 data.")
    label = "class"

    # Run AutoGluon
    l2_train_data = l2_X_train
    l2_train_data[label] = l2_y_train
    l2_test_data = l2_X_test
    l2_test_data[label] = l2_y_test

    import ray
    ray.init(local_mode=True)

    # Get constraints
    monotonic_constraints = [1 if col in oof_col_names else 0 for col in l2_train_data.columns if col != label]

    # Sample
    if sample is not None and (sample < len(l2_train_data)):
        l2_train_data = l2_train_data.sample(n=sample, random_state=0).reset_index(drop=True)
    if sample_test is not None and (sample_test < len(l2_test_data)):
        l2_test_data = l2_test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    predictor = TabularPredictor(eval_metric=eval_metric.name, label=label, verbosity=4).fit(
        train_data=l2_train_data,
        hyperparameters={
            # 'RF': [{'criterion': 'gini'}],
            'GBM':
                [
                    # {},
                    {
                        'monotone_constraints_for_stack_oof': True,
                        'monotone_constraints_method': 'basic', 'ag_args': {'name_suffix': '_monotonic'}
                    },
                ]
        },
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        num_bag_folds=8,

    )

    leaderboard_leak = predictor.leaderboard(l2_test_data, silent=False)

    exit()
    return
