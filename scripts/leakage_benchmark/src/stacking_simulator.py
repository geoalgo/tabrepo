from __future__ import annotations

from shutil import rmtree
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.metrics import Scorer, get_metric
from autogluon.features.generators import IdentityFeatureGenerator
from autogluon.tabular import TabularPredictor
from autogluon_zeroshot.repository import EvaluationRepository
from scripts.leakage_benchmark.src.config_and_data_utils import L1_PREFIX
from scripts.leakage_benchmark.src.custom_metadata_funcs import (
    _get_meta_data, _preprocessor_for_cluster, _sub_sample,
    get_l_fold_indicator)
from scripts.leakage_benchmark.src.other.distribution_insights import \
    get_proba_insights
from scripts.leakage_benchmark.src.other.post_hoc_ensembling import (
    caruana_weighted, roc_auc_binary_loss_proba)


def cluster_X(X):
    from sklearn.cluster import KMeans

    _clusterer = KMeans(n_clusters=len(X) // 20)
    _cluster_pre = _preprocessor_for_cluster()
    cluster_i = _clusterer.fit_predict(_cluster_pre.fit_transform(X))
    return cluster_i


def obtain_input_data_for_l2(
    repo: EvaluationRepository, l1_models: List[str], dataset: str, fold: int
) -> Tuple[pd.DataFrame, np.array, pd.DataFrame, np.array, Scorer, List[str], pd.DataFrame, FeatureMetadata]:
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

    problem_type = task_ground_truth_metadata["problem_type"]
    metric_name = task_ground_truth_metadata["eval_metric"]
    eval_metric = get_metric(metric=metric_name, problem_type=problem_type)
    eval_metric.problem_type = problem_type

    # - Obtain X and y
    # - Obtain X
    train_data, test_data = repo.get_data(tid, fold)
    l2_X_train, l2_y_train, l2_X_test, l2_y_test, l1_feature_metadata = repo.preprocess_data(tid, fold, train_data, test_data, reset_index=True)

    # Previous code had `y_test.fillna(-1)` in code. Not sure why, hence see where this happens with the assert.
    assert l2_y_test.hasnans is False
    l2_y_train = l2_y_train.to_numpy()
    l2_y_test = l2_y_test.fillna(-1).to_numpy()

    # - Obtain preds and build stack_X_train, stack_X_test
    pred_val, pred_test = repo._tabular_predictions.predict(dataset=tid, fold=fold, splits=["val", "test"], models=l1_models, force_1d=problem_type == "binary")
    oof_col_names = []
    classes = task_ground_truth_metadata["ordered_class_labels_transformed"]

    for i, m in enumerate(l1_models):

        if problem_type in ["binary", "regression"]:
            pred_val_m = pred_val[i]
            pred_test_m = pred_test[i]
            col_name = f"{L1_PREFIX}{m}"

            l2_X_train[col_name] = pred_val_m
            l2_X_test[col_name] = pred_test_m
            oof_col_names.append(col_name)
        elif problem_type in ["multiclass"]:

            preds_val_m = pred_val[i]
            preds_test_m = pred_test[i]
            col_names = [f"{L1_PREFIX}{m}/c{c_name}" for c_name in classes]

            l2_X_train[col_names] = preds_val_m
            l2_X_test[col_names] = preds_test_m
            oof_col_names.extend(col_names)
        else:
            raise NotImplementedError(f"Problem type {problem_type} not supported.")
    l2_X_train = l2_X_train.copy()
    l2_X_test = l2_X_test.copy()
    # Get L1 scores
    leaderboard = zsc.df_results_by_dataset_vs_automl.loc[
        (zsc.df_results_by_dataset_vs_automl["dataset"] == task) & (zsc.df_results_by_dataset_vs_automl["framework"].isin(l1_models)),
        ["framework", "metric_error", "score_val"],
    ]
    leaderboard["metric_error"] = leaderboard["metric_error"].apply(lambda x: eval_metric.optimum - x)
    leaderboard = leaderboard.rename(columns={"framework": "model", "metric_error": "score_test"}).reset_index(drop=True)

    return l2_X_train, l2_y_train, l2_X_test, l2_y_test, eval_metric, oof_col_names, classes, leaderboard, l1_feature_metadata


def _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata):
    # Follow autogluon/core/src/autogluon/core/models/ensemble/stacker_ensemble_model.py _add_stack_to_feature_metadata
    #   -- No additional preprocessing from our side here. If at all, this happens at the model level.
    from autogluon.common.features.types import R_FLOAT, S_STACK

    type_map_raw = {column: R_FLOAT for column in oof_col_names}
    type_group_map_special = {S_STACK: oof_col_names}
    stacker_feature_metadata = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)
    l2_feature_metadata = l1_feature_metadata.join_metadata(stacker_feature_metadata)

    return l2_feature_metadata


def get_duplicate_mask_and_sample_weights(l2_train_data, ignore_cols, label):
    print('Duplicate mask & sample weights calculation...')
    # Equalize code
    rel_cols = [x for x in l2_train_data.columns if x not in ignore_cols]
    keep_idx_list = []
    sample_weights = []

    for group_idx_list in l2_train_data.groupby(rel_cols).groups.values():
        # --- Sample weight and clever drop code
        group_idx_list = list(group_idx_list)
        n_dup = len(group_idx_list)
        if n_dup == 1:
            keep_index = group_idx_list[0]
            sample_count = 1
        else:
            sample_count = n_dup

            # Keep the majority label (or random if tie)
            subset = l2_train_data.loc[group_idx_list, label]
            counts = subset.value_counts()
            sel_label = counts.index[0]

            keep_index = subset[subset == sel_label].index[0]

        keep_idx_list.append(keep_index)
        sample_weights.append(sample_count)

    print('Done.')

    return keep_idx_list, sample_weights

def autogluon_l2_runner(
    l2_models,
    l2_X_train,
    l2_y_train,
    l2_X_test,
    l2_y_test,
    eval_metric: Scorer,
    oof_col_names: List[str],
    l1_feature_metadata: FeatureMetadata,
    get_meta_data: bool = False,
    sub_sample_data: bool = False,
    problem_type: str | None = None,
    classes=None,
    debug: bool = False,
    plot_insights: bool = False,
    l1_model_worst_to_best: List[str] = None,
    clear_save_path: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    print(f"Start preprocessing L2 data and collect metadata. {l2_X_train.shape} {l2_X_test.shape}")
    label = "class"
    l2_feature_metadata = _get_l2_feature_metadata(l2_X_train, l2_y_train, oof_col_names, l1_feature_metadata)

    # TODO: test performance of something like this as additional feature even without the leak
    #   Following wolpert, the idea is that l2 models learn from knowing the distance to the nearest neighbor
    #   Downside is that it is quite expensive I guess. (add at fit time not here)
    # _compute_nearest_neighbor_distance(l2_X_train.drop(columns=oof_col_names), l2_y_train,
    #                                    l2_X_test.drop(columns=oof_col_names))

    # _plot_problematic_instances(l2_X_train[[f'{L1_PREFIX}RandomForest_c1_BAG_L1']], l2_y_train)

    # Build data
    l2_train_data = l2_X_train
    l2_train_data[label] = l2_y_train
    l2_test_data = l2_X_test
    l2_test_data[label] = l2_y_test

    if get_meta_data:
        custom_meta_data = _get_meta_data(l2_train_data, l2_test_data, label, oof_col_names, problem_type, eval_metric)
    else:
        custom_meta_data = {}

    if sub_sample_data:
        l2_train_data, l2_test_data = _sub_sample(l2_train_data, l2_test_data)


    # Some tests
    from cir_model import CenteredIsotonicRegression
    from sklearn._isotonic import (_inplace_contiguous_isotonic_regression,
                                   _make_unique)
    from sklearn.base import clone
    from sklearn.calibration import _SigmoidCalibration
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss
    from .custom_sklearn._bagging import BaggingRegressor
    from sklearn.model_selection import cross_val_score

    ignore_feature_duplicates = oof_col_names + [label]
    # mask = ~l2_train_data.drop(columns=ignore_feature_duplicates).duplicated()
    #   keep_idx_list, sample_weight = get_duplicate_mask_and_sample_weights(l2_train_data, ignore_feature_duplicates, label)
    # l2_train_data[f] = reg.fit(l2_train_data.loc[keep_idx_list, [f]], l2_train_data.loc[keep_idx_list, label], sample_weight=sample_weight).predict(l2_train_data[[f]])


    from cir_model import CenteredIsotonicRegression
    # Run AutoGluon
    leaderboard_leak = None
    for cal_method in [
        'none',
        # ('_ir', False, IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
        # ('_cir', False, CenteredIsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
        # ('_sig', False, _SigmoidCalibration()),
        # ('_ir_p', True, IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
        # ('_cir_p',True, CenteredIsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
        # ('_sig_p',True, _SigmoidCalibration()),
        # ('_cust', False, None),
        # ('_cust_p', True, None),
        # ('_cust', None, None),
        # ('_cust_switch', None, 'switch'),

        # -- Old
        # ('_is_t', False, CenteredIsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
        # ('_is_b', False, BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True), n_estimators=100, random_state=42)),
        # ('_bis_p', True, BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True), n_estimators=100, random_state=42)),
        # ('_is_b-t', False, BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True, test=True), n_estimators=100, random_state=42)),
        # ('_is_b_d0.5', False, BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True), n_estimators=100, dropout=0.5, random_state=42)),
        # ('_bis_d0.1', False,BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True), n_estimators=100, dropout=0.1,  random_state=42)),
        # ('_bis_d0.9', False, BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True), n_estimators=100, dropout=0.9, random_state=42)),
        # ('_is_p', True, IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
        # ('_bis_d_p', True, BaggingRegressor(IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True), n_estimators=100, dropout=True, random_state=42)),
       ]:
        train_data = l2_train_data.copy()
        test_data = l2_test_data.copy()
        ir_map = {}
        if cal_method == 'none':
            postfix = ""
        else:
            postfix, test_apply, base_reg = cal_method
            print(f"Start Isotonic Regression Updates for {postfix}")
            if postfix.startswith('_cust'):
                for f in oof_col_names:
                    if base_reg is None:
                        if f.replace(L1_PREFIX, '') in ['NeuralNetTorch_c2_BAG_L1', 'NeuralNetTorch_c4_BAG_L1', 'NeuralNetTorch_c1_BAG_L1', 'NeuralNetTorch_c3_BAG_L1', 'NeuralNetFastAI_c1_BAG_L1']:
                            reg = _SigmoidCalibration()
                            test_apply = True if test_apply is None else test_apply
                        else:
                            reg = IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)
                            test_apply = False if test_apply is None else test_apply

                        # new_train_data = cross_val_predict(reg, train_data[[f]], train_data[label], cv=cv)
                        new_train_data = reg.fit(train_data[[f]], train_data[label]).predict(train_data[[f]])
                        train_data[f] = new_train_data
                        ir_map[f] = reg
                        if test_apply:
                            test_data[f] = reg.predict(test_data[[f]])
                    else:
                        cals = [
                            ('IR', False, IsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True)),
                            ('SIG', True, _SigmoidCalibration()),
                            ('CIR', False,CenteredIsotonicRegression(y_max=1, y_min=0, out_of_bounds='clip', increasing=True))
                        ]

                        res = []
                        best = None
                        def scorer(est, X, y):
                            return brier_score_loss(y, est.predict(X))
                        for cal_name, cal_test_apply, cal_reg in cals:
                            new_train_data_cal = cal_reg.fit(train_data[[f]], train_data[label]).predict(train_data[[f]])
                            new_test_data_cal = cal_reg.predict(test_data[[f]])
                            train_score = np.mean(cross_val_score(clone(cal_reg), X=train_data[[f]], y=train_data[label], scoring=scorer, cv=8))
                            # train_score = brier_score_loss(train_data[label], new_train_data_cal)
                            test_score = brier_score_loss(test_data[label], new_test_data_cal)
                            res.append((cal_name, cal_reg, cal_test_apply, new_train_data_cal, new_test_data_cal, train_score, test_score))
                            if (best is None) or (res[-1][-2] < res[-2][-2]):
                                best = res[-1]

                        # switch based on val data per fold maybe?
                        print(f"{f}-Train: " + " | ".join([f"{res_vals[0]} - {res_vals[-2]:.3f}" for res_vals in res]))
                        print(f"{f}-Test: " + " | ".join([f"{res_vals[0]} - {res_vals[-1]:.3f}" for res_vals in res]))
                        print(f"Best: {best[0]}")
                        train_data[f] = best[3]
                        ir_map[f] = best[1]
                        if best[2]:
                            test_data[f] = best[4]


            elif problem_type == 'binary':
                # from sklearn.model_selection import cross_val_predict, RepeatedStratifiedKFold

                #  rng = 123451
                # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=rng)
                for f in oof_col_names:
                    reg = clone(base_reg)
                    # new_train_data = cross_val_predict(reg, train_data[[f]], train_data[label], cv=cv)
                    new_train_data = reg.fit(train_data[[f]], train_data[label]).predict(train_data[[f]])
                    train_data[f] = new_train_data
                    ir_map[f] = reg
                    if test_apply:
                        test_data[f] = reg.predict(test_data[[f]])

            if False:
                X_train_org = l2_train_data
                X_train = train_data
                y_train = train_data[label]
                stack_cols = oof_col_names
                if ir_map is not None:
                    n = len(y_train)
                    fig, ax = plt.subplots(ncols=len(stack_cols), figsize=(20, 6))

                    for f_idx, f in enumerate(stack_cols):
                        ax[f_idx].plot(X_train_org[f], y_train, "C0.", markersize=12)
                        ax[f_idx].plot(X_train_org[f], X_train[f], "C2.",  markersize=12)
                        if hasattr(ir_map[f], "X_thresholds_"):
                            ax[f_idx].plot(ir_map[f].X_thresholds_, ir_map[f].y_thresholds_, "C1.-", markersize=12, alpha=0.5)
                        ax[f_idx].set_title(f)
                        ax[f_idx].set_xlim(-0.1, 1.1)

                    fig.supxlabel('Proba L1')
                    fig.supylabel('Label / Adjusted Proba')
                    plt.show()

        # from sklearn.calibration import CalibrationDisplay
        # fig, ax = plt.subplots(figsize=(10, 10))
        # colors = plt.get_cmap("tab20")
        # colors.colors = colors.colors + plt.get_cmap("tab20b").colors
        # colors.colors = colors.colors + plt.get_cmap("tab20c").colors
        # colors.colors = colors.colors + plt.get_cmap("Paired").colors
        # colors.N = len(colors.colors)
        #
        # for i, f in enumerate(oof_col_names):
        #     y = train_data[label].copy().astype(str)
        #
        #     if f.split("/")[-1][0] == 'c':
        #         curr_class = f.split("/")[-1][1:]
        #         pos_class_mask = y == curr_class
        #         y[pos_class_mask] = 1
        #         y[~pos_class_mask] = 0
        #         y = y.astype(int)
        #     else:
        #         curr_class = "1"
        #     display = CalibrationDisplay.from_predictions(y, train_data[f], n_bins=20, name=f+"/"+curr_class, ax=ax, color=colors(i))
        # plt.title(postfix)
        # plt.show()
        # continue

        features = list(train_data)
        for i_f_set in range(2):
            train_data = train_data[features]
            test_data = test_data[features]
            print(f"Start running AutoGluon on L2 data. Debug={debug}")
            try:
                predictor = TabularPredictor(eval_metric=eval_metric.name, label=label, verbosity=0, problem_type=problem_type, learner_kwargs=dict(random_state=1))
                fit_para = dict(ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}) if debug else dict()
                predictor.fit(
                    train_data=train_data,
                    hyperparameters=l2_models,
                    fit_weighted_ensemble=True,
                    num_stack_levels=0,
                    num_bag_folds=8,
                    feature_generator=IdentityFeatureGenerator(),
                    feature_metadata=l2_feature_metadata,
                    ag_args_fit=dict(),
                    **fit_para,
                )

                if plot_insights:
                    l2_train_oof = pd.DataFrame(
                        np.array([predictor.get_oof_pred_proba(m, as_multiclass=False).values for m in predictor.get_model_names()]).T,
                        columns=predictor.get_model_names(),
                    )
                    l2_test_oof = pd.DataFrame(
                        np.array([predictor.predict_proba(test_data, model=m, as_multiclass=False, as_pandas=False) for m in predictor.get_model_names()]).T,
                        columns=predictor.get_model_names(),
                    )

                    f_cols = [x for x in train_data.columns if x not in oof_col_names + [label]]
                    get_proba_insights(
                        train_data.loc[:, oof_col_names],
                        train_data.loc[:, oof_col_names],
                        l2_train_oof,
                        l2_test_oof,
                        train_data[label],
                        test_data[label],
                        train_data.loc[:, f_cols],
                        test_data.loc[:, f_cols],
                        eval_metric,
                        predictor,
                    )

                leaderboard_leak_i = predictor.leaderboard(test_data, silent=True)[["model", "score_test", "score_val"]]
                leaderboard_leak_i["model"] = leaderboard_leak_i["model"].apply(lambda x: x.replace("L1", "L2"))
                leaderboard_leak_i["model"] += postfix  + f'-feature-set-{i_f_set}'
                f_imp = predictor.feature_importance(test_data, num_shuffle_sets=100, features=list(set(oof_col_names).intersection(set(list(test_data)))))
                with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                    print(f_imp)
                features = list(set(f_imp[~f_imp.index.isin(f_imp['importance'].nlargest(5).index)].index))  + [label]
            finally:
                if clear_save_path:
                    print("Clean Up Path...")
                    rmtree(predictor.path)

            # print(leaderboard_leak_i)
            leaderboard_leak = leaderboard_leak_i if leaderboard_leak is None else pd.concat([leaderboard_leak, leaderboard_leak_i])


        # leaderboard_leak = pd.DataFrame(['-', 0,  0], columns=["model", "score_test", "score_val"])

    # # exit()
    # leaderboard_leak = leaderboard_leak[~leaderboard_leak.model.isin(['LightGBM_BAG_L2_is', 'LightGBM_BAG_L2_is_b', 'LightGBM_ir_BAG_L2',
    #                                                                   'LightGBM_ir_BAG_L2_is', 'LightGBM_ir_BAG_L2_is_b', 'LightGBM_ir_p_vp-full_BAG_L2_is', 'LightGBM_ir_p_vp-full_BAG_L2_is_p',
    #                                                                   'LightGBM_ir_BAG_L2_is_p', 'LightGBM_ir_p_BAG_L2_is', 'LightGBM_ir_p_BAG_L2'])]
    return leaderboard_leak, custom_meta_data



# Multiclass calibration
#             else:
#                 # --- Multiclass
#                 stack_cols = oof_col_names
#                 classes_in_X = set([f.rsplit("/")[-1][1:] for f in stack_cols])
#                 base_models = set([f.rsplit("/", 1)[0] for f in stack_cols])
#                 bm_proba_map = {bm: [f for f in stack_cols if f.startswith(bm)] for bm in base_models}
#                 y = train_data[label].copy()
#
#                 for bm, bm_stack_cols in bm_proba_map.items():
#                     c_ordered = np.unique(list(classes_in_X))
#                     f_ordered = np.unique(bm_stack_cols)
#
#                     for current_class, current_f in zip(c_ordered, f_ordered):
#                         try:
#                             y_ovr = y.copy().astype(int).astype(str)
#                         except:
#                             y_ovr = y.copy().astype(str)
#
#                         pos_class_mask = y_ovr == str(current_class)
#                         y_ovr[pos_class_mask] = 1
#                         y_ovr[~pos_class_mask] = 0
#                         y_ovr = y_ovr.astype(int)
#
#                         reg = base_reg
#                         train_data[current_f] = reg.fit_transform(train_data[current_f], y_ovr)
#                         if test_apply:
#                             test_data[current_f] = reg.predict(test_data[current_f])
#
#                     # Normalize the probabilities for multiclass following sklearn's implementation
#                     full_proba = train_data[f_ordered].values
#                     denominator = np.sum(full_proba, axis=1)[:, np.newaxis]
#                     uniform_proba = np.full_like(full_proba, 1 / len(c_ordered))
#                     full_proba = np.divide(full_proba, denominator, out=uniform_proba, where=denominator != 0)
#                     full_proba[(1.0 < full_proba) & (full_proba <= 1.0 + 1e-5)] = 1.0
#                     train_data[f_ordered] = full_proba
#
#                     if test_apply:
#                         full_proba = test_data[f_ordered].values
#                         denominator = np.sum(full_proba, axis=1)[:, np.newaxis]
#                         uniform_proba = np.full_like(full_proba, 1 / len(c_ordered))
#                         full_proba = np.divide(full_proba, denominator, out=uniform_proba, where=denominator != 0)
#                         full_proba[(1.0 < full_proba) & (full_proba <= 1.0 + 1e-5)] = 1.0
#                         test_data[f_ordered] = full_proba
#
#
#             # for f in oof_col_names:
#             #     curr_class = f.split("/")[-1][1:]
#             #     y = train_data[label].copy().astype(str)
#             #     pos_class_mask = y == curr_class
#             #     y[pos_class_mask] = 1
#             #     y[~pos_class_mask] = 0
#             #     y = y.astype(int)
#             #     reg = base_reg
#             #     train_data[f] = reg.fit(train_data[[f]], y).predict(train_data[[f]])\