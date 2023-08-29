import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_fold_indicator(X_train, y_train, layer):
    from autogluon.core.utils.utils import CVSplitter

    cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=layer)

    l1_fold_indicator = np.full((len(X_train),), np.nan)
    for fold_i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
        l1_fold_indicator[test_index] = fold_i

    return l1_fold_indicator


def _viz_flip(l1_oof, l2_oof, y, title_postfix='', l1_model='L1/OOF/RandomForest_c1_BAG_L1',
              l2_model='LightGBM_BAG_L2', l2_model_no_leak='LightGBM_lfc_BAG_L2'):
    plot_df = l1_oof.loc[:, [l1_model]]
    plot_df['label'] = y
    plot_df[l2_model] = l2_oof[l2_model]
    plot_df[l2_model_no_leak] = l2_oof[l2_model_no_leak]
    plot_df['fold'] = 'Train'
    plot_df['error'] = abs(l1_oof[l1_model] - l2_oof[l2_model])

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=plot_df['error'],
                  showscale=True,
                  colorscale='bluered'),
        dimensions=[
            dict(label=l1_model, range=[0, 1], values=plot_df[l1_model]),
            dict(label='label', range=[0, 1], values=plot_df['label']),
            dict(label=l2_model, range=[0, 1], values=plot_df[l2_model]),
            dict(label=l2_model_no_leak, range=[0, 1], values=plot_df[l2_model_no_leak]),
            dict(label='error', range=[0, 1], values=plot_df['error'])
        ],

    ), layout=dict(title=f'Label Flipping Insight{title_postfix}'))

    fig.show()


def _flip_model_check(X_train, y_train, X_test, y_test, l1_test_oof, l2_test_oof,
                      l1_model, l2_model, predictor, eval_metric):
    from sklearn.model_selection import cross_val_score
    from lightgbm import LGBMClassifier

    def bad_flips(label, l1_oof, l2_oof):
        l1_loss = abs(label - l1_oof)
        l2_loss = abs(label - l2_oof)
        return (l2_loss - l1_loss) >= 0

    # flip_X_train = pd.concat([X_train, l1_train_oof, l2_train_oof], axis=1)
    # flip_y_train = bad_flips(y_train, l1_train_oof[l1_model].values, l2_train_oof[l2_model].values)

    tmp_X_train = X_train.reset_index(drop=True).sample(n=min(len(X_train), 1000), random_state=42)
    tmp_y_train = y_train.iloc[tmp_X_train.index]
    tmp_y_train.reset_index(drop=True, inplace=True)
    tmp_X_train.reset_index(drop=True, inplace=True)

    # Add synthetic data
    proba_scales = np.arange(0, 1.01, 0.01)
    s_list = []
    for p_s in proba_scales:
        curr_scale = pd.Series(np.full(len(tmp_X_train), p_s), name=l1_model)

        l2_X = pd.concat([tmp_X_train, curr_scale], axis=1)
        l2_preds = pd.Series(
            predictor.predict_proba(l2_X, model=l2_model, as_multiclass=False, as_pandas=False),
            name=l2_model
        )
        mask = pd.Series(bad_flips(tmp_y_train, curr_scale.values, l2_preds.values), name='label')
        s_list.append(pd.concat([l2_X, l2_preds, mask], axis=1))

    synth_flip_data = pd.concat(s_list, axis=0)
    synth_flip_X_train = synth_flip_data.drop(columns=['label'])
    synth_flip_y_train = synth_flip_data['label'].values

    flip_X_train = synth_flip_X_train  # pd.concat([flip_X_train, synth_flip_X_train])
    flip_y_train = synth_flip_y_train  # np.hstack([flip_y_train, synth_flip_y_train])

    sample_goal = np.unique(flip_y_train, return_counts=True)[1][1]
    flip_X_train['class_label'] = flip_y_train
    flip_X_train = flip_X_train.groupby('class_label').sample(n=sample_goal, random_state=42).reset_index(drop=True)
    flip_y_train = flip_X_train['class_label'].values
    flip_X_train = flip_X_train.drop(columns=['class_label'])

    model_val_error = np.mean(cross_val_score(LGBMClassifier(random_state=42), flip_X_train, flip_y_train, cv=8,
                                            scoring='balanced_accuracy'))

    def flip_pred(_X, _y, _X_test, _to_flip_oof, _baseline_oof):
        flip_chart = LGBMClassifier(random_state=42).fit(_X, _y).predict_proba(_X_test)[:, 1]
        flip_mask = flip_chart >= 0.2
        test_oof_tmp = _to_flip_oof.copy()
        test_oof_tmp.loc[flip_mask] = _baseline_oof.loc[flip_mask]
        return test_oof_tmp, flip_mask

    flip_X_test = pd.concat([X_test, l1_test_oof, l2_test_oof], axis=1)
    # Test Score
    new_y_pred, flip_mask = flip_pred(flip_X_train, flip_y_train, flip_X_test, l2_test_oof[l2_model],
                                      l1_test_oof[l1_model])
    score = eval_metric(y_test, new_y_pred)
    flip_ratio = sum(flip_mask) / len(flip_mask)
    return score, flip_ratio, model_val_error


def get_proba_insights(l1_train_oof, l1_test_oof, l2_train_oof, l2_test_oof, y_train, y_test,
                       X_train, X_test, eval_metric, predictor):
    # hue as unique yes or no

    # 100 ros

    # FLIP CHECKER
    # incrementally increase from 0 to 1 over 1l probas and get diff to l2 probas
    # with original features and for subset of instances
    # get mean squared error or absolute error

    # Algo
    # need features and predictor for this

    # l1_fold_indicator = get_fold_indicator(l1_train_oof, y_train, layer=1)
    #
    # print('l1')
    # for bm in list(l1_train_oof):
    #     val_score = eval_metric(y_train, l1_train_oof[bm])
    #     test_score = eval_metric(y_test, l1_test_oof[bm])
    #
    #     print(bm, test_score, val_score)
    #
    # print('l2')
    # for bm in list(l2_train_oof):
    #     val_score = eval_metric(y_train, l2_train_oof[bm])
    #     test_score = eval_metric(y_test, l2_test_oof[bm])
    #
    #     print(bm, test_score, val_score)
    #
    # l1_model = list(l1_train_oof)[0]
    # l2_model = list(l2_train_oof)[0]
    #
    # # # Get Val Score (technically leaks due to predictor I guess)
    # # from autogluon.core.utils.utils import CVSplitter
    # # cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=2)
    # # res, res_, res__ = [], [], []
    # # for train_index, test_index in cv.split(X_train, y_train):
    # #     score, flip_ratio, model_val_error = _flip_model_check(X_train.iloc[train_index], y_train[train_index],
    # #                                           X_train.iloc[test_index], y_train[test_index],
    # #                                           l1_train_oof.iloc[test_index], l2_train_oof.iloc[test_index],
    # #                                           l1_model, l2_model, predictor, eval_metric)
    # #     res.append(score)
    # #     res_.append(flip_ratio)
    # #     res__.append(model_val_error)
    # # print(f'CV Avoided Score: {np.mean(res)} (FR: {np.mean(res_)}) | Model Val Error: {np.mean(res__)}')
    #
    #
    # # Get Test Score
    # score, flip_ratio, model_val_error = _flip_model_check(X_train, y_train, X_test, y_test, l1_test_oof, l2_test_oof,
    #                                       l1_model, l2_model, predictor, eval_metric)
    # print(f'Flip Avoided Test Score: {score} (FR: {flip_ratio}) | Model Val Score: {model_val_error}')

    # --- 1 Base Model Viz ---
    l2_train_oof.columns = [x.replace('L1', 'L2') for x in l2_train_oof.columns]
    l2_test_oof.columns = [x.replace('L1', 'L2') for x in l2_test_oof.columns]
    _viz_flip(l1_train_oof, l2_train_oof, y_train, title_postfix=' for Train')
    _viz_flip(l1_test_oof, l2_test_oof, y_test, title_postfix=' for Test')
    input('Press enter to continue')
