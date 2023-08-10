from __future__ import annotations

import numpy as np
import pandas as pd


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


def _plot_problematic_instances(X, y):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    tree = DecisionTreeClassifier().fit(X, y)
    plot_tree(tree,
              feature_names=tree.feature_names_in_,
              filled=True,
              rounded=True)
    plt.savefig("tree.pdf")
    pd.DataFrame(tree.apply(X)).groupby(by=0)[0].count().mean()

    df = pd.DataFrame(tree.apply(X))
    df[1] = y
    df.groupby(by=0)[1].mean().mean()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    tree = DecisionTreeClassifier(min_samples_leaf=5).fit(X, y)
    plot_tree(tree,
              feature_names=tree.feature_names_in_,
              filled=True,
              rounded=True)
    plt.savefig("tree_2.pdf")
    exit(0)


def _preprocess_save_for_sklearn(X_train, y_train, X_test):
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector
    from sklearn.preprocessing import OrdinalEncoder
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
        ]
    )
    X_train = pd.DataFrame(preprocessor.fit_transform(X_train, y_train), columns=X_train.columns)
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=X_test.columns)

    return X_train, X_test


def _get_leaf_node_view(X_train, y_train, X_test, y_test, min_samples_leaf, problem_type, oof_col_names):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from functools import partial
    import math
    from decimal import localcontext

    X_train, X_test = _preprocess_save_for_sklearn(X_train, y_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if problem_type == 'regression':
        tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
    else:
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

    tree.fit(X_train, y_train)
    group_function = tree.apply

    def misguided_ratio(subset_df, l1_accuracy):
        # This won't work for regression or multi-class without defining alpha and beta in a different way
        sc = subset_df.value_counts()
        alpha = sc['a'] if 'a' in sc else 0
        beta = sc['b'] if 'b' in sc else 0
        p = alpha + beta

        with localcontext() as ctx:
            ctx.prec = 32
            f_a = math.factorial(alpha)
            f_b = math.factorial(beta)
            f_p = math.factorial(p)
            save_fact_ratio = float(ctx.divide(f_p, ctx.multiply(f_a, f_b)))

        assert math.isfinite(save_fact_ratio),  "save_fact_ratio is not a finite number"

        data_proba_l1_is_incorrect = 1 - (l1_accuracy)**alpha * (1 - l1_accuracy)**beta * save_fact_ratio
        potential_cheat_instances = beta * data_proba_l1_is_incorrect

        return potential_cheat_instances

    def stats(X, y):
        df = pd.DataFrame(group_function(X))
        df[1] = y

        avg_ = []
        for oof_col in oof_col_names:
            tmp_oof_col = X[oof_col].copy()
            threshold = 0.5
            pseudo_correct_mask = ((tmp_oof_col > threshold) & (y == 1)) | ((tmp_oof_col <= threshold) & (y == 0))

            tmp_oof_col[pseudo_correct_mask] = 'a'
            tmp_oof_col[~pseudo_correct_mask] = 'b'
            accuracy = sum(pseudo_correct_mask)/len(pseudo_correct_mask)

            df[3] = tmp_oof_col
            potential_for_cheat_ratio = df.groupby(by=0)[3].apply(partial(misguided_ratio, l1_accuracy=accuracy)).sum()/len(X)
            avg_.append(potential_for_cheat_ratio)

        return dict(avg_rel_sample_count=(df.groupby(by=0)[0].count() / len(X)).mean(),
                    avg_potential_for_cheat_ratio=avg_)

    return dict(train_stats=stats(X_train, y_train),
                test_stats=stats(X_test, y_test))


def _get_leaf_duplicated_view(X_train, y_train, X_test, y_test, oof_col_names):
    from functools import partial
    import math

    X_train, X_test = _preprocess_save_for_sklearn(X_train, y_train, X_test)

    def misguided_count(subset_df):
        sc = subset_df.value_counts()
        alpha = sc['a'] if 'a' in sc else 0
        beta = sc['b'] if 'b' in sc else 0
        if beta > alpha:
            return beta
        return 0


    def stats(X, y):
        avg_ = []
        for oof_col in oof_col_names:
            tmp_oof_col = X[oof_col].copy()
            df = pd.DataFrame(tmp_oof_col)
            threshold = 0.5
            pseudo_correct_mask = ((tmp_oof_col > threshold) & (y == 1)) | ((tmp_oof_col <= threshold) & (y == 0))
            tmp_oof_col[pseudo_correct_mask] = 'a'
            tmp_oof_col[~pseudo_correct_mask] = 'b'
            df[1] = tmp_oof_col

            potential_for_cheat_ratio = df.groupby(oof_col)[1].apply(partial(misguided_count)).sum()/len(X)
            avg_.append(potential_for_cheat_ratio)

        return dict(avg_potential_for_cheat_ratio=avg_)

    return dict(train_stats=stats(X_train, y_train),
                test_stats=stats(X_test, y_test))


def _cv_wrapper_avg_cheat(X_train, y_train, min_samples_leaf, problem_type, oof_col_names):
    from autogluon.core.utils.utils import CVSplitter

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=1)

    i_dict_list = []
    for train_index, test_index in cv.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
        i_dict_list.append(_get_leaf_node_view(X_train_cv, y_train_cv, X_test_cv, y_test_cv,
                                               min_samples_leaf=min_samples_leaf, problem_type=problem_type,
                                               oof_col_names=oof_col_names))
    return pd.concat([pd.DataFrame(x) for x in i_dict_list]).groupby(
        level=0).mean().rename(columns={'test_stats': 'val_stats'}).to_dict()


def _all_wrong_count(X, y, oof_col_names, threshold=0.5):
    stack_f = oof_col_names
    tmp_X = X[stack_f].copy()
    classes_ = np.unique(y)
    tmp_X = tmp_X.mask(tmp_X <= threshold, classes_[0])
    tmp_X = tmp_X.mask(tmp_X > threshold, classes_[1])
    s_tmp = tmp_X.sum(axis=1)

    no_diversity_rows = (s_tmp == 0) | (s_tmp == len(stack_f))
    s_tmp = s_tmp[no_diversity_rows]
    s_tmp[s_tmp == len(stack_f)] = 1

    return sum(s_tmp != y[no_diversity_rows])/len(X)