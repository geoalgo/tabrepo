from __future__ import annotations

import pandas as pd
import numpy as np

from typing import List, Tuple, Dict

from autogluon.core.metrics import get_metric, Scorer
from autogluon_zeroshot.repository import EvaluationRepository
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator, AutoMLPipelineFeatureGenerator
from autogluon.common.features.feature_metadata import FeatureMetadata

from scripts.leakage_benchmark.src.config_and_data_utils import L1_PREFIX


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
    eval_metric.problem_type = problem_type

    # - Obtain X and y
    # - Obtain X
    train_data, test_data = repo.get_data(tid, fold)
    l2_X_train, l2_y_train, l2_X_test, l2_y_test, l1_feature_metadata \
        = repo.preprocess_data(tid, fold, train_data, test_data, reset_index=True)

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
        col_name = f'{L1_PREFIX}{m}'

        l2_X_train[col_name] = pred_val_m
        l2_X_test[col_name] = pred_test_m
        oof_col_names.append(col_name)

    # Get L1 scores
    leaderboard = zsc.df_results_by_dataset_vs_automl.loc[(zsc.df_results_by_dataset_vs_automl['dataset'] == task) & (
        zsc.df_results_by_dataset_vs_automl['framework'].isin(l1_models)), ['framework', 'metric_error', 'score_val']]
    leaderboard['metric_error'] = leaderboard['metric_error'].apply(lambda x: eval_metric.optimum - x)
    leaderboard = leaderboard.rename(columns={'framework': 'model', 'metric_error': 'score_test'}).reset_index(
        drop=True)

    return l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names, leaderboard, l1_feature_metadata


def _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata):
    prepr = AutoMLPipelineFeatureGenerator()
    prepr.fit_transform(l2_X_train[oof_col_names], l2_y_train)

    l2_feature_metadata = l1_feature_metadata.join_metadata(prepr.feature_metadata)

    # Add stack metadata
    l2_feature_metadata.type_group_map_special['stack'] = oof_col_names

    return l2_feature_metadata


def _sub_sample(l2_train_data, l2_test_data, sample=20000, sample_test=10000):
    # Sample
    if sample is not None and (sample < len(l2_train_data)):
        l2_train_data = l2_train_data.sample(n=sample, random_state=0).reset_index(drop=True)
    if sample_test is not None and (sample_test < len(l2_test_data)):
        l2_test_data = l2_test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    return l2_train_data, l2_test_data


def _compute_nearest_neighbor_distance(X_train, y_train, X_test):
    # Compute nearest neighbor distance
    from autogluon.core.utils.utils import CVSplitter
    from sklearn.neighbors import NearestNeighbors
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    preprocessor = Pipeline(
        [
            ('fix', ColumnTransformer(transformers=[
                ("num", SimpleImputer(strategy="constant", fill_value=-1),
                 make_column_selector(dtype_exclude="object"),),
                ("cat",
                 Pipeline(steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),),
                                 ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                                 ]), make_column_selector(dtype_include="object"),), ], sparse_threshold=0, )),
            ('scale', StandardScaler())
        ]
    )
    X_train = X_train.reset_index(drop=True)
    cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=1)

    oof_distance = np.full_like(y_train, np.nan)
    test_distances = []

    for train_index, test_index in cv.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]

        X_train_cv = preprocessor.fit_transform(X_train_cv)
        X_test_cv = preprocessor.transform(X_test_cv)
        nn_m = NearestNeighbors(n_neighbors=1)
        nn_m.fit(X_train_cv)
        oof_distance[test_index] = nn_m.kneighbors(X_test_cv)[0].reshape(-1)

        test_distances.append(nn_m.kneighbors(preprocessor.transform(X_test))[0].reshape(-1))

    print("n+duplicates:", sum(pd.DataFrame(preprocessor.fit_transform(X_train)).duplicated()) / len(X_train))
    return oof_distance, np.mean(np.array(test_distances), axis=0)


def _find_optimal_threshold(y, proba):
    from sklearn.metrics import balanced_accuracy_score

    threshold_pos = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                     0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def proba_to_binary(proba, t):
        return np.where(proba >= t, 1, 0)

    tf = threshold_pos[
        np.argmax(
            [
                balanced_accuracy_score(
                    y, proba_to_binary(proba, ti)
                )
                for ti in threshold_pos
            ]
        )
    ]
    return tf


def autogluon_l2_runner(l2_models, l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric: Scorer,
                        oof_col_names: List[str], l1_feature_metadata: FeatureMetadata,
                        sub_sample_data: bool = False, problem_type: str | None = None) -> Tuple[pd.DataFrame, Dict]:
    print("Start running AutoGluon on L2 data.")
    label = "class"
    l2_feature_metadata = _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata)

    # TODO: test performance of something like this as additional feature even without the leak
    #   Following wolpert, the idea is that l2 models learn from knowing the distance to the nearest neighbor
    #   Downside is that it is quite expensive I guess. (add at fit time not here)
    # _compute_nearest_neighbor_distance(l2_X_train.drop(columns=oof_col_names), l2_y_train,
    #                                    l2_X_test.drop(columns=oof_col_names))

    # Build data
    l2_train_data = l2_X_train
    l2_train_data[label] = l2_y_train
    l2_test_data = l2_X_test
    l2_test_data[label] = l2_y_test

    # Compute metadata
    f_dup = oof_col_names + [label]
    f_l_dup = oof_col_names

    custom_meta_data = dict(
        train_l2_duplciates=sum(l2_train_data.duplicated()) / len(l2_train_data),
        train_feature_duplciates=sum(l2_train_data.drop(columns=f_dup).duplicated()) / len(l2_train_data),
        train_feature_label_duplicates=sum(l2_train_data.drop(columns=f_l_dup).duplicated()) / len(l2_train_data),
        test_l2_duplicates=sum(l2_test_data.duplicated()) / len(l2_test_data),
        test_feature_duplciates=sum(l2_test_data.drop(columns=f_dup).duplicated()) / len(l2_test_data),
        test_feature_label_duplicates=sum(l2_test_data.drop(columns=f_l_dup).duplicated()) / len(l2_test_data),

        # Unique
        train_unique_vlaues_per_oof=[(col, len(np.unique(l2_train_data[col]))) for col in oof_col_names],
        test_unique_vlaues_per_oof=[(col, len(np.unique(l2_test_data[col]))) for col in oof_col_names],
    )

    if problem_type == 'binary':
        custom_meta_data['optimal_threshold_train_per_oof'] = \
            [(col, _find_optimal_threshold(l2_train_data[label], l2_train_data[col])) for col in oof_col_names]
        custom_meta_data['optimal_threshold_test_per_oof'] = \
            [(col, _find_optimal_threshold(l2_test_data[label], l2_test_data[col])) for col in oof_col_names]

    if sub_sample_data:
        l2_train_data, l2_test_data = _sub_sample(l2_train_data, l2_test_data)

    # import ray
    # ray.init(local_mode=True)

    # Run AutoGluon
    predictor = TabularPredictor(eval_metric=eval_metric.name, label=label, verbosity=0, problem_type=problem_type,
                                 learner_kwargs=dict(random_state=1)).fit(
        train_data=l2_train_data,
        hyperparameters=l2_models,
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        num_bag_folds=8,
        feature_generator=IdentityFeatureGenerator(),
        feature_metadata=l2_feature_metadata

    )

    leaderboard_leak = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']]
    leaderboard_leak['model'] = leaderboard_leak['model'].apply(lambda x: x.replace('L1', 'L1.5'))

    return leaderboard_leak, custom_meta_data
