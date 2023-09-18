from __future__ import annotations

import numpy as np
import pandas as pd
import ray


def _sub_sample(l2_train_data, l2_test_data, sample=20000, sample_test=10000):
    # Sample
    if sample is not None and (sample < len(l2_train_data)):
        l2_train_data = l2_train_data.sample(n=sample, random_state=0).reset_index(drop=True)
    if sample_test is not None and (sample_test < len(l2_test_data)):
        l2_test_data = l2_test_data.sample(n=sample_test, random_state=0).reset_index(drop=True)

    return l2_train_data, l2_test_data


def _get_ag_cv(layer):
    from autogluon.core.utils.utils import CVSplitter

    cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=layer)
    return cv


def _compute_nearest_neighbor_distance(X_train, y_train, X_test):
    # Compute nearest neighbor distance
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    from autogluon.core.utils.utils import CVSplitter

    preprocessor = Pipeline(
        [
            (
                "fix",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            SimpleImputer(strategy="constant", fill_value=-1),
                            make_column_selector(dtype_exclude="object"),
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    (
                                        "encoder",
                                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                    ),
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                                ]
                            ),
                            make_column_selector(dtype_include="object"),
                        ),
                    ],
                    sparse_threshold=0,
                ),
            ),
            ("scale", StandardScaler()),
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

    threshold_pos = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def proba_to_binary(proba, t):
        return np.where(proba >= t, 1, 0)

    tf = threshold_pos[np.argmax([balanced_accuracy_score(y, proba_to_binary(proba, ti)) for ti in threshold_pos])]
    return tf


def _plot_problematic_instances(X, y):
    import matplotlib.pyplot as plt
    from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                              plot_tree)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    tree = DecisionTreeClassifier().fit(X, y)
    plot_tree(tree, feature_names=tree.feature_names_in_, filled=True, rounded=True)
    plt.savefig("tree.pdf")
    pd.DataFrame(tree.apply(X)).groupby(by=0)[0].count().mean()

    df = pd.DataFrame(tree.apply(X))
    df[1] = y
    df.groupby(by=0)[1].mean().mean()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    tree = DecisionTreeClassifier(min_samples_leaf=5).fit(X, y)
    plot_tree(tree, feature_names=tree.feature_names_in_, filled=True, rounded=True)
    plt.savefig("tree_2.pdf")
    exit(0)


def _preprocess_save_for_sklearn(X_train, y_train, X_test):
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    preprocessor = Pipeline(
        [
            (
                "fix",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            SimpleImputer(strategy="constant", fill_value=-1),
                            make_column_selector(dtype_exclude="object"),
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    (
                                        "encoder",
                                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                    ),
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                                ]
                            ),
                            make_column_selector(dtype_include="object"),
                        ),
                    ],
                    sparse_threshold=0,
                ),
            ),
        ]
    )
    X_train = pd.DataFrame(preprocessor.fit_transform(X_train, y_train), columns=X_train.columns)
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=X_test.columns)

    return X_train, X_test


def _get_leaf_node_view(X_train, y_train, X_test, y_test, min_samples_leaf, problem_type, oof_col_names, p_small_threshold=10):
    import math
    from decimal import localcontext
    from functools import partial

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    X_train, X_test = _preprocess_save_for_sklearn(X_train, y_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if problem_type == "regression":
        tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
    else:
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)

    tree.fit(X_train, y_train)
    group_function = tree.apply

    def misguided_ratio(subset_df, l1_accuracy):
        # This won't work for regression or multi-class without defining alpha and beta in a different way
        sc = subset_df.value_counts()
        alpha = sc["a"] if "a" in sc else 0
        beta = sc["b"] if "b" in sc else 0
        p = alpha + beta

        if (p <= p_small_threshold) or (beta > alpha):
            p1 = (l1_accuracy) ** alpha * (1 - l1_accuracy) ** beta

            if p1 > 0:
                with localcontext() as ctx:
                    ctx.prec = 32
                    f_a = math.factorial(alpha)
                    f_b = math.factorial(beta)
                    f_p = math.factorial(p)
                    save_fact_ratio = float(ctx.divide(f_p, ctx.multiply(f_a, f_b)))

                assert math.isfinite(save_fact_ratio), "save_fact_ratio is not a finite number"
                data_proba_l1_is_incorrect = 1 - p1 * save_fact_ratio
            else:
                data_proba_l1_is_incorrect = 0
            potential_cheat_instances = beta * data_proba_l1_is_incorrect
        else:
            potential_cheat_instances = 0

        return potential_cheat_instances

    def stats(X, y):
        df = pd.DataFrame(group_function(X))
        df[1] = y

        avg_ = []
        for oof_col in oof_col_names:
            tmp_oof_col = X[oof_col].copy()
            threshold = 0.5
            pseudo_correct_mask = ((tmp_oof_col > threshold) & (y == 1)) | ((tmp_oof_col <= threshold) & (y == 0))

            tmp_oof_col[pseudo_correct_mask] = "a"
            tmp_oof_col[~pseudo_correct_mask] = "b"
            accuracy = sum(pseudo_correct_mask) / len(pseudo_correct_mask)

            df[3] = tmp_oof_col
            potential_for_cheat_ratio = df.groupby(by=0)[3].apply(partial(misguided_ratio, l1_accuracy=accuracy)).sum() / len(X)
            avg_.append(potential_for_cheat_ratio)

        return dict(avg_rel_sample_count=(df.groupby(by=0)[0].count() / len(X)).mean(), avg_potential_for_cheat_ratio=avg_)

    return dict(train_stats=stats(X_train, y_train), test_stats=stats(X_test, y_test))


def _get_leaf_duplicated_view(X_train, y_train, X_test, y_test, oof_col_names):
    import math
    from functools import partial

    X_train, X_test = _preprocess_save_for_sklearn(X_train, y_train, X_test)

    def misguided_count(subset_df):
        sc = subset_df.value_counts()
        alpha = sc["a"] if "a" in sc else 0
        beta = sc["b"] if "b" in sc else 0
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
            tmp_oof_col[pseudo_correct_mask] = "a"
            tmp_oof_col[~pseudo_correct_mask] = "b"
            df[1] = tmp_oof_col

            potential_for_cheat_ratio = df.groupby(oof_col)[1].apply(partial(misguided_count)).sum() / len(X)
            avg_.append(potential_for_cheat_ratio)

        return dict(avg_potential_for_cheat_ratio=avg_)

    return dict(train_stats=stats(X_train, y_train), test_stats=stats(X_test, y_test))


def _cv_wrapper_avg_cheat(X_train, y_train, min_samples_leaf, problem_type, oof_col_names):
    from autogluon.core.utils.utils import CVSplitter

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=1)

    i_dict_list = []
    for train_index, test_index in cv.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
        i_dict_list.append(
            _get_leaf_node_view(
                X_train_cv, y_train_cv, X_test_cv, y_test_cv, min_samples_leaf=min_samples_leaf, problem_type=problem_type, oof_col_names=oof_col_names
            )
        )
    return pd.concat([pd.DataFrame(x) for x in i_dict_list]).groupby(level=0).mean().rename(columns={"test_stats": "val_stats"}).to_dict()


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

    return sum(s_tmp != y[no_diversity_rows]) / len(X)


def to_label(proba):
    arr = proba.values

    if len(arr.shape) == 1:
        arr = np.vstack([1 - arr, arr]).T
    return np.argmax(arr, axis=1)


def _linkage_matrix(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, np.full_like(counts, np.nan), counts]).astype(float)

    return linkage_matrix


def _cluster_plot(model, **kwargs):
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    # Create linkage matrix and then plot the dendrogram
    # Plot the corresponding dendrogram
    dendrogram(_linkage_matrix(model), **kwargs)
    plt.show()


def _preprocessor_for_cluster():
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    preprocessor = Pipeline(
        [
            (
                "fix",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            SimpleImputer(strategy="constant", fill_value=-1),
                            make_column_selector(dtype_exclude="object"),
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    (
                                        "encoder",
                                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                    ),
                                    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                                ]
                            ),
                            make_column_selector(dtype_include="object"),
                        ),
                    ],
                    sparse_threshold=0,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    return preprocessor


def wrong_likelihood(data, label_col, oof_col_names, groups):
    from scipy.stats import binom
    from sklearn.metrics import accuracy_score

    global_acc_map = {oof_col: accuracy_score(data[label_col], to_label(data[oof_col])) for oof_col in oof_col_names}

    likelihood_correct_per_oof = []
    for group_i in groups:
        mask = groups == group_i
        subset = data[mask]

        likelihood_correct_for_subset_per_oof = []
        for oof_col in oof_col_names:
            alpha = sum(to_label(subset[oof_col]) == subset[label_col])
            likelihood_correct_for_subset_per_oof.append(binom.pmf(alpha, len(subset), global_acc_map[oof_col]))

        likelihood_correct_per_oof.append(likelihood_correct_for_subset_per_oof)

    return np.mean(np.mean(np.array(likelihood_correct_per_oof), axis=0))


def _value_expectation(new_oof, oof, y, i, lm, total_n_samples, _lambda, tol_range):
    left_i, right_i, distance, n_samples = lm[i]

    # FIXME, change lm array type instead of cast here
    left_i = int(left_i)
    right_i = int(right_i)

    if left_i < total_n_samples:
        left_i_list, left_sum = [left_i], oof[left_i]
    else:
        left_i_list, left_sum = _value_expectation(new_oof, oof, y, left_i - total_n_samples, lm, total_n_samples, _lambda, tol_range)

    if right_i < total_n_samples:
        right_i_list, right_sum = [right_i], oof[right_i]
    else:
        right_i_list, right_sum = _value_expectation(new_oof, oof, y, right_i - total_n_samples, lm, total_n_samples, _lambda, tol_range)

    all_i = left_i_list + right_i_list
    sum_vals = left_sum + right_sum

    left_i_exp = left_sum / len(left_i_list)
    right_i_exp = right_sum / len(right_i_list)
    level_exp = sum_vals / n_samples

    # l_shrinkage, r_shrinkage = (1 + _lambda / len(left_i_list)), (1 + _lambda / len(right_i_list))
    shrinkage = 1 + _lambda / n_samples
    # tol_range = 0.2
    new_oof[left_i_list] += (left_i_exp - level_exp) / shrinkage
    new_oof[right_i_list] += (right_i_exp - level_exp) / shrinkage

    return all_i, sum_vals


def hierarchical_oof_shrinkage(X, oof_col_names):
    from scipy.cluster.hierarchy import linkage

    lm = linkage(_preprocessor_for_cluster().fit_transform(X.drop(columns=oof_col_names)), metric="euclidean", optimal_ordering=False, method="ward")

    _lambda = 100
    for oof_col in oof_col_names:
        oof = X[oof_col].values
        new_oof = np.full_like(oof, np.mean(oof))
        _value_expectation(new_oof, oof, -1, lm, len(oof), _lambda)
        X[oof_col] = new_oof


def _get_cheat_likelihood(data: pd.DataFrame, label_col: str, oof_col_names: list[str], eval_metric) -> float:
    from scipy.cluster.hierarchy import linkage
    from sklearn.cluster import AgglomerativeClustering

    cluster_model = AgglomerativeClustering(n_clusters=50, compute_distances=False)
    cluster_model = cluster_model.fit(_preprocessor_for_cluster().fit_transform(data.drop(columns=[label_col])))
    lm = _linkage_matrix(cluster_model)
    # lm = linkage(_preprocessor_for_cluster().fit_transform(data.drop(columns=oof_col_names + [label_col])),
    #              metric='euclidean', optimal_ordering=False, method='ward')

    _lambda = 100

    for oof_col in oof_col_names:
        oof = data[oof_col].values
        new_oof = np.full_like(oof, np.mean(oof))
        _value_expectation(new_oof, oof, data[label_col], -1, lm, len(oof), _lambda, 0)
        data[oof_col] = new_oof
        # print(np.mean(abs(oof - new_oof)))

        # print(f'\n ### {oof_col}')
        # print(eval_metric(data[label_col], data[oof_col]))
        #
        # best_oof = oof.copy()
        # best_score = eval_metric(data[label_col], oof)
        # for _lambda in [1, 5, 25, 50, 100, 500]:
        #     new_oof = np.full_like(oof, np.mean(oof))
        #     _value_expectation(new_oof, oof, -1, lm, len(oof), _lambda)
        #
        #     if eval_metric(data[label_col], new_oof) > best_score:
        #         best_oof = new_oof.copy()
        #
        # print(eval_metric(data[label_col], oof), eval_metric(data[label_col], best_oof))
        # data[oof_col] = best_oof
        # # print(eval_metric(data[label_col], data[oof_col]))

    return 0
    exit(-1)
    print()

    # groups = cluster_model.fit(_preprocessor_for_cluster().fit_transform(data.drop(columns=[label_col])))

    print(
        wrong_likelihood(
            data,
            label_col,
            oof_col_names,
            groups
            # data.groupby(list(data.columns)).ngroup()
        )
    )

    # from sklearn.metrics import accuracy_score

    #
    # global_acc_map = {oof_col: accuracy_score(data[label_col], to_label(data[oof_col])) for oof_col in oof_col_names}
    # [np.mean(data[oof_col]) for oof_col in oof_col_names]
    #
    # # Get acc over clusters

    #
    # _cluster_plot(cluster_model, truncate_mode="level", p=3)
    # _cluster_plot(cluster_model, truncate_mode="level", p=30)
    # def subset_func(subset_df):
    #     print()
    #
    # data.groupby(list(data.columns)).apply(subset_func)
    return 0


def _weight_vector_merge_exp():
    # -- Weight Vector merger exp
    from collections import Counter

    def _calculate_final_weights(indices_, num_input_models_):
        ensemble_members = Counter(indices_).most_common()
        weights = np.zeros(
            (num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / len(indices_)
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        return weights

    # Idea merge OOF predictions into new vectors that obscures the leak without losing information
    new_oofs = []
    # rng = np.random.RandomState(42)
    for idx, oof_col in enumerate(oof_col_names):
        p = np.ones(len(oof_col_names))
        p[idx] += 1
        p = p / p.sum()

        # # per row
        # x = _calculate_final_weights(rng.choice(len(oof_col_names), len(oof_col_names), replace=True, p=p),
        #                          len(oof_col_names))
        # rng.choice(len(oof_col_names), len(oof_col_names), replace=True, p=p)
        X_train[oof_col] = np.average(X_train[oof_col_names], axis=1, weights=p)
        X_test[oof_col] = np.average(X_test[oof_col_names], axis=1, weights=p)


def oof_melt(
    train_oof_df,
    test_oof_df,
    fold_indicator,
    value_name,
):
    oof_df = train_oof_df.copy()
    oof_df_p2 = test_oof_df.copy()

    oof_df.loc[:, "fold"] = "Fold " + pd.Series(fold_indicator.astype(int).astype(str))
    oof_df = oof_df.melt(id_vars=["fold"], var_name="model", value_name=value_name)

    oof_df_p2.loc[:, "fold"] = "Test"
    oof_df_p2 = oof_df_p2.melt(id_vars=["fold"], var_name="model", value_name=value_name)
    oof_df = pd.concat([oof_df, oof_df_p2])
    return oof_df


def _dist_overview_plot(train_oof_df, test_oof_df, fold_indicator, models_to_plot):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    value_name = "Observed Prediction Values"
    order = [f"Fold {int(i)}" for i in range(int(max(fold_indicator)) + 1)] + ["Test"]

    oof_df = oof_melt(train_oof_df, test_oof_df, fold_indicator, value_name)

    plot_df = oof_df[oof_df["model"].isin(models_to_plot)]
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(9, rot=-0.25, light=0.7)
    pal[-1] = [1, 0, 0]
    g = sns.FacetGrid(plot_df, row="fold", hue="fold", aspect=15, height=1, palette=pal, row_order=order, hue_order=order)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, value_name, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, value_name, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, value_name)
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.suptitle(f"Distribution of predictions")
    return g


def _stack_hist_plot(train_oof_df, test_oof_df, hue_indicator, models_to_plot):
    import matplotlib.pyplot as plt
    import seaborn as sns

    pal = sns.cubehelix_palette(9, rot=-0.25, light=0.7)
    pal[-1] = [1, 0, 0]
    value_name = "Observed Prediction Values"
    oof_df = oof_melt(train_oof_df, test_oof_df, hue_indicator, value_name)
    plot_df = oof_df[oof_df["model"].isin(models_to_plot)]

    sns.set_theme(style="ticks")

    f, ax = plt.subplots(figsize=(15, 8))
    ax.set_title(f"Overlap of Predictions")

    g = sns.histplot(
        data=plot_df,
        x=value_name,
        hue="fold",
        multiple="stack",
        palette=pal,
        edgecolor=".3",
        linewidth=0.5,
        ax=ax
        # log_scale=True,
    )
    return g


def _stack_hist_plot_facet(train_oof_df, y_train, test_oof_df, y_test, fold_indicator, models_to_plot):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    test_oof_df = test_oof_df.copy()
    train_oof_df.loc[:, "y"] = y_train
    test_oof_df = test_oof_df.copy()
    test_oof_df.loc[:, "y"] = y_test

    value_name = "Observed Prediction Values"
    order = [f"Fold {int(i)}" for i in range(int(max(fold_indicator)) + 1)] + ["Test"]

    oof_df = train_oof_df.copy()
    oof_df_p2 = test_oof_df.copy()

    oof_df.loc[:, "fold"] = "Fold " + pd.Series(fold_indicator.astype(int).astype(str))
    oof_df = oof_df.melt(id_vars=["fold", "y"], var_name="model", value_name=value_name)

    oof_df_p2.loc[:, "fold"] = "Test"
    oof_df_p2 = oof_df_p2.melt(id_vars=["fold", "y"], var_name="model", value_name=value_name)
    oof_df = pd.concat([oof_df, oof_df_p2])

    plot_df = oof_df[oof_df["model"].isin(models_to_plot)]

    g = sns.displot(
        plot_df,
        x=value_name,
        row="y",
        col="fold",
        height=3,
        aspect=0.9,
        facet_kws=dict(margin_titles=True),
        col_order=order,
        kde=True,
    )

    return g


def _plot_prediction_distributions(train_oof_df, y_train, test_oof_df, y_test, fold_indicator):
    import matplotlib.pyplot as plt

    models = ["L1/OOF/NeuralNetFastAI_c1_BAG_L1", "L1/OOF/CatBoost_c1_BAG_L1", "L1/OOF/RandomForest_c1_BAG_L1"]  # list(train_oof_df)

    for m in models:
        print(m)
        # _stack_hist_plot(train_oof_df, test_oof_df, fold_indicator, [m])
        # _dist_overview_plot(train_oof_df, test_oof_df, fold_indicator, [m])
        _stack_hist_plot_facet(train_oof_df, y_train, test_oof_df, y_test, fold_indicator, [m])
        plt.show()

    return


from shutil import rmtree

from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.metrics import accuracy_score

from autogluon.tabular import TabularPredictor


@ray.remote
def run_fold(X_test, y_test, train_X, train_y, val_X, val_y, eval_metric, problem_type, fold_i, oof_col_names):
    predictor = TabularPredictor(eval_metric=eval_metric.name, label="label", verbosity=0, problem_type=problem_type, learner_kwargs=dict(random_state=1))
    # predictor_2 = TabularPredictor(eval_metric=eval_metric.name, label="label", verbosity=0, problem_type=problem_type, learner_kwargs=dict(random_state=1))

    l2_train_data = train_X.copy()
    l2_train_data["label"] = train_y
    l2_val_data = val_X.copy()
    l2_val_data["label"] = val_y

    # TODO: can we just fit without validation data (if the model does not need tuning data for something like early stopping or if we can disable early stopping)
    predictor.fit(
        train_data=l2_train_data,
        tuning_data=l2_val_data,
        hyperparameters={"GBM": [{}]},
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        num_bag_folds=0,
        holdout_frac=0,
        num_cpus=1,
        num_gpus=0,
    )
    predictor.calibrate_decision_threshold(metric="accuracy")

    # predictor_2.fit(
    #     train_data=l2_train_data.drop(columns=oof_col_names),
    #     tuning_data=l2_val_data.drop(columns=oof_col_names),
    #     hyperparameters={"GBM": [{}]},
    #     fit_weighted_ensemble=False,
    #     num_stack_levels=0,
    #     num_bag_folds=0,
    #     holdout_frac=0,
    #     num_cpus=1,
    #     num_gpus=0,
    # )
    # predictor_2.calibrate_decision_threshold(metric="accuracy")

    val_score = eval_metric(val_y, predictor.predict_proba(val_X).values[:, 1])
    test_score = eval_metric(y_test, predictor.predict_proba(X_test).values[:, 1])
    #acc_val_score = accuracy_score(val_y, predictor.predict(val_X).values)
    #acc_test_score = accuracy_score(y_test, predictor.predict(X_test).values)

    # l_val = len(val_X)
    # rng = np.random.RandomState(fold_i)
    # leak_repo_scores = []
    # random_scores = []
    # no_oof_scores = []
    # for _ in range(10):  # fixme> could I generalize this to OOF data?
    #     shuffle_oof = train_X[oof_col_names].sample(frac=1.0, random_state=rng)
    #
    #     while len(shuffle_oof):
    #         iter_val_X = val_X.copy().reset_index(drop=True)
    #
    #         tmp_shuffle_oof = shuffle_oof.iloc[:l_val]
    #         shuffle_oof = shuffle_oof.drop(index=tmp_shuffle_oof.index)
    #
    #         if len(tmp_shuffle_oof) < l_val:
    #             tmp_shuffle_oof = pd.concat([tmp_shuffle_oof, train_X[oof_col_names].sample(n=l_val - len(tmp_shuffle_oof), random_state=rng)])
    #
    #         iter_val_X[oof_col_names] = tmp_shuffle_oof.reset_index(drop=True)
    #         iter_pred = predictor.predict(iter_val_X).values
    #         iter_no_oof_pred = predictor_2.predict(iter_val_X.drop(columns=oof_col_names)).values
    #
    #         pred_correct_mask = val_y.values == iter_pred
    #         pred_no_oof_correct_mask = val_y.values == iter_no_oof_pred
    #         rnd_correct_mask = val_y.values == rng.choice(np.unique(val_y), len(val_y), replace=True)
    #
    #         # tmp_shuffle_index = tmp_shuffle_oof.index.copy().values
    #         # label_equal_mask = train_y[tmp_shuffle_index].values == val_y.values
    #
    #         leak_repo_scores.append(np.mean(pred_correct_mask))
    #         random_scores.append(np.mean(rnd_correct_mask))
    #         no_oof_scores.append(np.mean(pred_no_oof_correct_mask))


    # leak_repo_score = np.mean(leak_repo_scores)
    # random_score = np.mean(random_scores)
    # no_oof_score = np.mean(no_oof_scores)
    rmtree(predictor.path)
    # rmtree(predictor_2.path)
    return [val_score, test_score]


def _leak_reproduction_score(X_train, y_train, X_test, y_test, label, oof_col_names, problem_type, eval_metric):
    futures = [
        run_fold.remote(
            X_test,
            y_test,
            X_train.iloc[train_index],
            y_train.iloc[train_index],
            X_train.iloc[test_index],
            y_train.iloc[test_index],
            eval_metric,
            problem_type,
            fold_i,
            oof_col_names,
        )
        for fold_i, (train_index, test_index) in enumerate(_get_ag_cv(layer=2).split(X_train, y_train))
    ]

    res = []
    for fold_res in ray.get(futures):
        res.append(fold_res)

    res = np.mean(np.array(res), axis=0)
    print(f"ROC: Val: {res[0]}, Test: {res[3]} | Acc: Val: {res[4]}, Test: {res[5]} | Leak Repo Acc: {res[1]}  | Random Leak Repo Acc: {res[2]} | No OOF Acc: {res[-1]} ")

    return res


def get_l_fold_indicator(X_train, y_train, layer, stratify_on=None,stratify_on_last_layer=False):
    l1_fold_indicator = np.full((len(X_train),), np.nan)
    stratify_on_vals = y_train if stratify_on is None else stratify_on
    for fold_i, (train_index, test_index) in enumerate(_get_ag_cv(layer=layer).split(X_train,

                                                                                     get_l_fold_indicator(X_train, stratify_on_vals, layer-1) if stratify_on_last_layer else stratify_on_vals)):
        l1_fold_indicator[test_index] = fold_i
    return l1_fold_indicator

def _explore_val_vs_pred(X_train, y_train, X_test, y_test, label, oof_col_names, problem_type, eval_metric):
    import matplotlib.pyplot as plt

    l1_fold_indicator = np.full((len(X_train),), np.nan)
    for fold_i, (train_index, test_index) in enumerate(_get_ag_cv(layer=1).split(X_train, y_train)):
        l1_fold_indicator[test_index] = fold_i

    _dist_overview_plot(X_train[oof_col_names], X_test[oof_col_names], l1_fold_indicator, ["L1/OOF/LightGBM_c1_BAG_L1"])

    l1_fold_indicator = np.full((len(X_train),), np.nan)
    for fold_i, (train_index, test_index) in enumerate(_get_ag_cv(layer=2).split(X_train, y_train)):
        l1_fold_indicator[test_index] = fold_i

    _dist_overview_plot(X_train[oof_col_names], X_test[oof_col_names], l1_fold_indicator, ["L1/OOF/LightGBM_c1_BAG_L1"])
    # _plot_prediction_distributions(X_train[oof_col_names].copy(), y_train, X_test[oof_col_names].copy(), y_test, l1_fold_indicator)

    # with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
    #     print(X_train[oof_col_names].describe() - X_test[oof_col_names].describe())
    plt.show()

    return


def _get_meta_data(l2_train_data, l2_test_data, label, oof_col_names, problem_type, eval_metric):
    # Init
    f_dup = oof_col_names + [label]
    f_l_dup = oof_col_names
    train_n_instances = len(l2_train_data)
    n_columns = len(l2_train_data.columns)
    test_n_instances = len(l2_test_data)

    X_train = l2_train_data.drop(columns=[label])
    y_train = l2_train_data[label]
    X_test = l2_test_data.drop(columns=[label])
    y_test = l2_test_data[label]
    to_drop_cols = _leak_reproduction_score(X_train, y_train, X_test, y_test, label, oof_col_names, problem_type, eval_metric)
    return {"tdc": to_drop_cols}

    _explore_val_vs_pred(X_train, y_train, X_test, y_test, label, oof_col_names, problem_type, eval_metric)

    # print('train', _get_cheat_likelihood(l2_train_data, label, oof_col_names, eval_metric))
    # print('test', _get_cheat_likelihood(l2_test_data, label, oof_col_names, eval_metric))
    return {}
    # Compute metadata
    custom_meta_data = dict(
        train_l2_duplicates=sum(l2_train_data.duplicated()) / train_n_instances,
        train_feature_duplicates=sum(l2_train_data.drop(columns=f_dup).duplicated()) / train_n_instances,
        test_feature_duplicates=sum(l2_test_data.drop(columns=f_dup).duplicated()) / test_n_instances,
        test_l2_duplicates=sum(l2_test_data.duplicated()) / test_n_instances,
        test_feature_label_duplicates=sum(l2_test_data.drop(columns=f_l_dup).duplicated()) / test_n_instances,
        train_feature_label_duplicates=sum(l2_train_data.drop(columns=f_l_dup).duplicated()) / train_n_instances,
        train_duplicated_columns=sum(l2_train_data.T.duplicated()) / n_columns,
        test_duplicated_columns=sum(l2_test_data.T.duplicated()) / n_columns,
        # Unique
        train_unique_values_per_oof=[len(np.unique(l2_train_data[col])) / train_n_instances for col in oof_col_names],
        test_unique_values_per_oof=[len(np.unique(l2_test_data[col])) / test_n_instances for col in oof_col_names],
        # Basic properties
        train_n_instances=train_n_instances,
        test_n_instances=test_n_instances,
        n_columns=n_columns,
        problem_type=problem_type,
        eval_metric_name=eval_metric.name,
        oof_col_names_order=oof_col_names,
    )

    if problem_type == "binary":
        custom_meta_data["optimal_threshold_train_per_oof"] = [_find_optimal_threshold(l2_train_data[label], l2_train_data[col]) for col in oof_col_names]
        custom_meta_data["optimal_threshold_test_per_oof"] = [_find_optimal_threshold(l2_test_data[label], l2_test_data[col]) for col in oof_col_names]
        custom_meta_data["always_wrong_row_ratio_train"] = _all_wrong_count(
            X_train, y_train, oof_col_names, threshold=np.mean(custom_meta_data["optimal_threshold_train_per_oof"])
        )
        custom_meta_data["always_wrong_row_ratio_test"] = _all_wrong_count(
            X_test, y_test, oof_col_names, threshold=np.mean(custom_meta_data["optimal_threshold_test_per_oof"])
        )

        custom_meta_data["potential_for_cheat_stats_tree_view"] = _get_leaf_node_view(
            X_train, y_train, X_test, y_test, min_samples_leaf=1, problem_type=problem_type, oof_col_names=oof_col_names
        )

        custom_meta_data["potential_for_cheat_stats_duplicates_view"] = _get_leaf_duplicated_view(X_train, y_train, X_test, y_test, oof_col_names=oof_col_names)
        # custom_meta_data['potential_for_cheat_stats_cv'] = \
        #     _cv_wrapper_avg_cheat(X_train, y_train, min_samples_leaf=1, problem_type=problem_type, oof_col_names=oof_col_names )

    return custom_meta_data
