import pandas as pd
import numpy as np
from scripts.leakage_benchmark.src.holdout_based_solutions.ag_test_utils import get_best_val_models

def loss_per_sample(y_true, y_pred):
    from sklearn.preprocessing import LabelBinarizer
    y_pred = y_pred.values
    y_true = y_true.values

    lb = LabelBinarizer()
    lb.fit(y_true)
    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )
    # Clipping
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = (transformed_labels * (transformed_labels - y_pred)).sum(axis=1)  # np.log(y_pred)
    return loss

def tuner_tests(leaderboard, predictor, train_data, test_data):
    best_l1_model, best_l2_model, _ = get_best_val_models(leaderboard)
    label = predictor.label

    oof_proba_no_leak = predictor.get_oof_pred_proba(best_l1_model)
    val_perf_no_leak = predictor.evaluate_predictions(train_data[label], oof_proba_no_leak, auxiliary_metrics=False)
    loss_ps_no_leak = loss_per_sample(train_data[label], oof_proba_no_leak)
    oof_proba_no_leak.columns = ["NL" + x for x in oof_proba_no_leak]
    oof_proba_leak = predictor.get_oof_pred_proba(best_l2_model)
    val_perf_leak = predictor.evaluate_predictions(train_data[label], oof_proba_leak, auxiliary_metrics=False)
    loss_ps_leak = loss_per_sample(train_data[label], oof_proba_leak)
    oof_proba_leak.columns = ["L" + x for x in oof_proba_leak]
    v = pd.concat([oof_proba_no_leak, oof_proba_leak, train_data[label], pd.Series(abs(loss_ps_no_leak - loss_ps_leak), name='L1-L2-Diff', index=train_data.index)], axis=1)

    test_proba_no_leak = predictor.predict_proba(test_data, model=best_l1_model)
    test_perf_no_leak = predictor.evaluate_predictions(test_data[label], test_proba_no_leak, auxiliary_metrics=False)
    test_loss_ps_no_leak = loss_per_sample(test_data[label], test_proba_no_leak)
    test_proba_no_leak.columns = ["NL" + x for x in test_proba_no_leak]
    test_proba_leak = predictor.predict_proba(test_data, model=best_l2_model)
    test_perf_leak = predictor.evaluate_predictions(test_data[label], test_proba_leak, auxiliary_metrics=False)
    test_loss_ps_leak = loss_per_sample(test_data[label], test_proba_leak)
    test_proba_leak.columns = ["L" + x for x in test_proba_leak]

    t = pd.concat([test_proba_no_leak, test_proba_leak, test_data[label], pd.Series(abs(test_loss_ps_leak - test_loss_ps_no_leak), name='L1-L2-Diff', index=test_data.index)], axis=1)


    from sklearn.isotonic import  IsotonicRegression
    
    # for F in
    IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip').fit()
    print()
