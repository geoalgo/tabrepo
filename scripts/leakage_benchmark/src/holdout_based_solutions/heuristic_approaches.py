from autogluon.core.metrics import get_metric
from autogluon.tabular import TabularPredictor
from scripts.leakage_benchmark.src.holdout_based_solutions.ag_test_utils import \
    get_best_val_models
from scripts.leakage_benchmark.src.holdout_based_solutions.logger import \
    get_logger

logger = get_logger()


def no_holdout(train_data, label, fit_para, predictor_para, **kwargs):
    method_name = "no_holdout_heuristic"

    logger.debug(f"Start running AutoGluon on data: {method_name}")
    predictor = TabularPredictor(**predictor_para)
    predictor.fit(train_data=train_data, **fit_para)

    problem_type = predictor_para["problem_type"]
    if problem_type in ["binary", "multiclass"]:
        alt_metric = "roc_auc"
    else:
        alt_metric = "mse"

    # Decide between L1 and L2 model based on heuristic
    leaderboard = predictor.leaderboard(silent=True)

    # Determine best l1 and l2 model
    best_l1_model, best_l2_model, leaking_models_exist = get_best_val_models(leaderboard)
    score_l1_oof = leaderboard.loc[leaderboard["model"] == best_l1_model, "score_val"].iloc[0]
    score_l2_oof = leaderboard.loc[leaderboard["model"] == best_l2_model, "score_val"].iloc[0]

    if (not leaking_models_exist) or (score_l1_oof >= score_l2_oof):
        return predictor, method_name, None

    # -- Obtain reproduction scores
    X = predictor.transform_features(train_data.drop(columns=[label]))
    y = predictor.transform_labels(train_data[label])

    l2_repo_oof = predictor.predict_proba(X, model=best_l2_model, as_multiclass=False, as_reproduction_predictions_args=dict(y=y))

    l1_models = [model_name for model_name in leaderboard["model"] if model_name.endswith("BAG_L1")]
    model_name_to_oof = {model_name: predictor.get_oof_pred_proba(model=model_name, as_multiclass=False) for model_name in l1_models}
    l2_true_repo_oof = predictor.predict_proba(
        X, model=best_l2_model, as_multiclass=False, as_reproduction_predictions_args=dict(y=y, model_name_to_oof=model_name_to_oof)
    )

    eval_metric = get_metric(alt_metric, problem_type=problem_type)
    am_score_l2_repo = eval_metric(y, l2_repo_oof)
    am_score_l2_true_repo = eval_metric(y, l2_true_repo_oof)

    # -- Detect leak and if we think the leak happens set best model to be the L1 model
    if am_score_l2_repo < am_score_l2_true_repo:
        logger.debug(f"Avoid leak with heuristic!")
        predictor.set_model_best(best_l1_model, save_trainer=True)

    return predictor, method_name, None
