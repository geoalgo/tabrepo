from shutil import rmtree

import numpy as np
import openml
import pandas as pd

from scripts.leakage_benchmark.src.holdout_based_solutions.logger import \
    get_logger

logger = get_logger()


def get_data(tid: int, fold: int):
    # Get Task and dataset from OpenML and return split data
    oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True, download_qualities=False, download_features_meta_data=False)

    train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
    X, *_ = oml_task.get_dataset().get_data(dataset_format="dataframe")

    return (
        X.iloc[train_ind, :].reset_index(drop=True),
        X.iloc[test_ind, :].reset_index(drop=True),
        oml_task.target_name,
        oml_task.task_type != "Supervised Classification",
    )


def sub_sample(l2_train_data, l2_test_data, label, n_max_cols, n_max_train_instances, n_max_test_instances):
    # Sub sample instances
    l2_train_data = l2_train_data.sample(n=min(len(l2_train_data), n_max_train_instances), random_state=0).reset_index(drop=True)
    l2_test_data = l2_test_data.sample(n=min(len(l2_test_data), n_max_test_instances), random_state=0).reset_index(drop=True)

    # Sub sample columns
    cols = list(l2_train_data.columns)
    cols.remove(label)
    if len(cols) > n_max_cols:
        cols = list(np.random.RandomState(42).choice(cols, replace=False, size=n_max_cols))
    l2_train_data = l2_train_data[cols + [label]]
    l2_test_data = l2_test_data[cols + [label]]

    logger.debug(f"{l2_train_data.shape}, {l2_test_data.shape}")
    return l2_train_data, l2_test_data, label


def get_best_val_models(leaderboard):
    leaderboard = leaderboard.copy()
    non_leaking = ["WeightedEnsemble_L2", "WeightedEnsemble_BAG_L2"]
    for non_leaker in non_leaking:
        leaderboard["model"] = leaderboard["model"].str.replace(non_leaker, non_leaker.replace("L2", "L1"))
    best_l1_model = leaderboard[leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]
    leaking_models_exist = any(m.endswith("L2") for m in leaderboard["model"])

    if leaking_models_exist:
        # Get best models per layer
        best_l2_model = leaderboard[~leaderboard["model"].str.endswith("L1")].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]
    else:
        best_l2_model = None

    # -- Revert back
    if best_l1_model in [x.replace("L2", "L1") for x in non_leaking]:
        best_l1_model = best_l1_model.replace("L1", "L2")
    for non_leaker in non_leaking:
        leaderboard["model"] = leaderboard["model"].str.replace(non_leaker.replace("L2", "L1"), non_leaker)

    return best_l1_model, best_l2_model, leaking_models_exist


def _check_stacked_overfitting_from_leaderboard(leaderboard):
    best_l1_model, best_l2_model, leaking_models_exist = get_best_val_models(leaderboard)

    score_l1_oof = leaderboard.loc[leaderboard["model"] == best_l1_model, "score_val"].iloc[0]
    score_l1_test = leaderboard.loc[leaderboard["model"] == best_l1_model, "score_test"].iloc[0]

    if leaking_models_exist:
        score_l2_oof = leaderboard.loc[leaderboard["model"] == best_l2_model, "score_val"].iloc[0]
        score_l2_test = leaderboard.loc[leaderboard["model"] == best_l2_model, "score_test"].iloc[0]

        # l1 worse val score than l2+
        stacked_overfitting = score_l1_oof < score_l2_oof
        # l2+ worse test score than L1
        stacked_overfitting = stacked_overfitting and (score_l1_test >= score_l2_test)

    else:
        # Stacked Overfitting is impossible
        score_l2_oof = np.nan
        score_l2_test = np.nan
        stacked_overfitting = False

    return stacked_overfitting, score_l1_oof, score_l2_oof, score_l1_test, score_l2_test


def inspect_leaderboard(leaderboard, final_model_name):
    logger.debug("### Leaderboard Summary")

    stacked_overfitting, score_l1_oof, score_l2_oof, score_l1_test, score_l2_test = _check_stacked_overfitting_from_leaderboard(leaderboard)

    # Final Score
    final_val_score = leaderboard.loc[leaderboard["model"] == final_model_name, "score_val"].iloc[0]
    final_test_score = leaderboard.loc[leaderboard["model"] == final_model_name, "score_test"].iloc[0]
    val_best_model = leaderboard.sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    logger.debug(f"L1 OOF: \t {score_l1_oof:.5f} | L1 Test: \t {score_l1_test:.5f}")
    logger.debug(f"L2 OOF: \t {score_l2_oof:.5f} | L2 Test: \t {score_l2_test:.5f}")
    logger.debug(f"Final Val: \t {final_val_score:.5f} | Final Test: \t {final_test_score:.5f}")

    # Check whether we selected the best model from all trained models
    final_is_best = final_test_score >= leaderboard["score_test"].max()

    logger.debug(f"Stacked Overfitting: {stacked_overfitting} | Final Model is Best: {final_is_best}")

    return dict(
        SO=stacked_overfitting,
        best_model=final_is_best,
        test_score=final_test_score,
        final_model=final_model_name,
        val_best_model=val_best_model,
    )


from sklearn.preprocessing import LabelEncoder


def print_and_get_leaderboard(predictor, l2_test_data, method_name, corrected_val_scores):
    l2_test_data = l2_test_data.copy()
    if hasattr(predictor, 'post_hoc_metric'):
        l2_test_data[predictor.label] = LabelEncoder().fit_transform(l2_test_data[predictor.label])
        predictor._learner.eval_metric = predictor.post_hoc_metric
        leaderboard = predictor.leaderboard(l2_test_data, silent=True)
    else:
        leaderboard = predictor.leaderboard(l2_test_data, silent=True)
    leaderboard = leaderboard[["model", "score_test", "score_val"]].sort_values(by="model").reset_index(
            drop=True)
    logger.debug(f"### Results for {method_name}")
    if corrected_val_scores is not None:
        leaderboard = leaderboard.merge(corrected_val_scores.rename({"score_test": "unbiased_score_val"}, axis=1), on="model")

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        logger.debug(leaderboard.sort_values(by="score_val", ascending=False))

    # # Verifiy best test score
    # from autogluon.core.metrics import get_metric
    # m = get_metric("roc_auc", problem_type="binary")
    # y_pred_proba = predictor.predict_proba(l2_test_data.drop(columns=predictor.label)).values
    # if hasattr(predictor, '_to_binary_proba'):
    #     y_pred_proba = predictor._to_binary_proba(y_pred_proba)
    # else:
    #     y_pred_proba = y_pred_proba[:, 1]
    #
    # print(f'Sanity Test Score: {m(LabelEncoder().fit_transform(l2_test_data[predictor.label]), y_pred_proba)}')

    rmtree(predictor.path)

    return leaderboard


def inspect_full_results(res_dict, proxy_so_results):
    logger.info("\n===> Task Results Summary")
    res_df = pd.DataFrame(res_dict["results"]).T
    res_df.index.name = "method_name"
    res_df = res_df.reset_index()
    logger.info(f"Proxy Found Stacked Overfitting: {proxy_so_results}")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        logger.info(res_df.sort_values(by="test_score", ascending=False))
