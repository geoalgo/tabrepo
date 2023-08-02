from __future__ import annotations

import pandas as pd
import numpy as np

from typing import List, Tuple

from autogluon.core.metrics import get_metric, Scorer
from autogluon_zeroshot.repository import EvaluationRepository
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator, AutoMLPipelineFeatureGenerator
from autogluon.common.features.feature_metadata import FeatureMetadata


def obtain_input_data_for_l2(repo: EvaluationRepository, l1_models: List[str], dataset: str, fold: int) \
        -> Tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, Scorer, List[str], pd.DataFrame, FeatureMetadata]:
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
    l2_X_train, l2_y_train, l2_X_test, l2_y_test, l1_feature_metadata \
        = repo.preprocess_data(tid, fold, train_data, test_data)

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
            raise NotImplementedError  # TODO support multiclass and regression here

        pred_val_m = pred_val[i]
        pred_test_m = pred_test[i]
        col_name = f'L1/OOF/{m}'

        l2_X_train[col_name] = pred_val_m
        l2_X_test[col_name] = pred_test_m
        oof_col_names.append(col_name)

    # Get L1 scores
    leaderboard = zsc.df_results_by_dataset_vs_automl.loc[(zsc.df_results_by_dataset_vs_automl['dataset'] == task) & (
        zsc.df_results_by_dataset_vs_automl['framework'].isin(l1_models)), ['framework', 'metric_error', 'score_val']]
    leaderboard['metric_error'] = leaderboard['metric_error'].apply(lambda x: eval_metric.optimum - x)
    leaderboard = leaderboard.rename(columns={'framework': 'model', 'metric_error': 'score_test'})

    return l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names, leaderboard, l1_feature_metadata


def _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata):
    prepr = AutoMLPipelineFeatureGenerator()
    prepr.fit_transform(l2_X_train[oof_col_names], l2_y_train)

    # TODO: could add stacking as special tpye group here if we need it later. But stick to this for now.
    l2_feature_metadata = l1_feature_metadata.join_metadata(prepr.feature_metadata)

    return l2_feature_metadata


def _sub_sample(l2_train_data, l2_test_data, sample=20000, sample_test=10000):
    # Sample
    if sample is not None and (sample < len(l2_train_data)):
        l2_train_data = l2_train_data.sample(n=sample, random_state=0).reset_index(drop=True)
    if sample_test is not None and (sample_test < len(l2_test_data)):
        l2_test_data = l2_test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    return l2_train_data, l2_test_data


def autogluon_l2_runner(l2_models, l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric: Scorer, oof_col_names: List[str],
                        l1_feature_metadata: FeatureMetadata, sub_sample_data: bool = False) -> pd.DataFrame:
    print("Start running AutoGluon on L2 data.")
    label = "class"
    l2_feature_metadata = _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata)

    # Run AutoGluon
    l2_train_data = l2_X_train
    l2_train_data[label] = l2_y_train
    l2_test_data = l2_X_test
    l2_test_data[label] = l2_y_test

    if sub_sample_data:
        l2_train_data, l2_test_data = _sub_sample(l2_train_data, l2_test_data)

    predictor = TabularPredictor(eval_metric=eval_metric.name, label=label, verbosity=1).fit(
        train_data=l2_train_data,
        hyperparameters=l2_models,
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        num_bag_folds=8,
        feature_generator=IdentityFeatureGenerator(),
        feature_metadata=l2_feature_metadata

    )

    leaderboard_leak = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']]
    leaderboard_leak['model'] = leaderboard_leak['model'].apply(lambda x: x.replace('L1', 'L2'))

    return leaderboard_leak
