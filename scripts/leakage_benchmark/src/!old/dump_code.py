# # Duplicate Code before L2 not during CV
# ignore_feature_duplicates = oof_col_names + [label]
# ignore_feature_label_duplicates = oof_col_names
# ignore_cols = ignore_feature_duplicates
# mask = l2_train_data.drop(columns=ignore_cols).duplicated()
# print("n+duplicates:", sum(mask) / len(mask))
#
# # Equalize code
# rel_cols = [x for x in l2_train_data.columns if x not in ignore_cols]
# for group_idx_list in l2_train_data.groupby(rel_cols).groups.values():
#
#     if len(group_idx_list) == 1:
#         continue
#     for oof_col in oof_col_names:
#         curr_vals = l2_train_data.iloc[group_idx_list, l2_train_data.columns.get_loc(oof_col)]
#         # could do majority or avg here... not sure what is better, stick to avg for now
#         l2_train_data.iloc[group_idx_list, l2_train_data.columns.get_loc(oof_col)] = curr_vals.max()
# print("Equalize done.")
#
# # Drop code
# # l2_train_data = l2_train_data[~mask]

# ---- Fold association code
#
# u_len = len(str(len(fold_fit_args_list)) * n_repeats) + n_repeats + 1
# layer_splits_indicator = np.full(len(X), "", dtype=f"U{u_len}")
# for fold_fit_args in fold_fit_args_list:
#     fold_fitting_strategy.schedule_fold_model_fit(**fold_fit_args)
#     fold_mask = fold_fit_args["fold_ctx"]["fold"][1]
#     layer_splits_indicator[fold_mask] = np.char.add(layer_splits_indicator[fold_mask],
#                                                     "S" + str(fold_fit_args["fold_ctx"]["split_index"]))
# self._layer_splits_indicator = layer_splits_indicator
# fold_fitting_strategy.after_all_folds_scheduled()
#
# if (level > 1) and (core_kwargs['ag_args_ensemble'].get("also_stratify_on_previous_layer", False)):
#     # TODO: ignores the fact that some models could crash right now...
#     split_indicators = self.get_models_attribute_dict('layer_split_indicator', base_model_names)
#     l_indicators = {k: len(np.unique(v)) for k, v in split_indicators.items() if v is not None}
#     if l_indicators:
#         core_kwargs['previous_layer_splits'] = split_indicators[min(l_indicators.items(), key=lambda x: x[1])[0]]
#
#
# # -- Associate data to bag
#         # from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap
#         # n_samples = y.shape[0]
#         # n_samples_bootstrap = _get_n_samples_bootstrap(
#         #     n_samples,
#         #     self.model.max_samples,
#         # )
#         # u_len = (len(str(len(self.model.estimators_)))+1)*len(self.model.estimators_) + 1  # could use 70% too
#         # layer_splits_indicator = np.full(len(X), "", dtype=f"U{u_len}")
#         # for est_i, estimator in enumerate(self.model.estimators_):
#         #     unsampled_indices = _generate_unsampled_indices(
#         #         estimator.random_state,
#         #         n_samples,
#         #         n_samples_bootstrap,
#         #     )
#         #     layer_splits_indicator[unsampled_indices] = np.char.add(layer_splits_indicator[unsampled_indices], "S" + str(est_i))
#         self._layer_splits_indicator = None # layer_splits_indicator
#         # --- End test



# --- Cluster indicator and rank per instance view
#     l1_fold_indicator = get_l_fold_indicator(l2_X_train, l2_y_train, layer=1)
#     l2_fold_indicator = get_l_fold_indicator(l2_X_train, l2_y_train, layer=2, stratify_on_last_layer=False)
# cluster_indicator = cluster_X(l2_X_train)
# l1_ges_weights, val_loss_over_iterations_ = caruana_weighted([np.array(x) for x in l2_train_data[oof_col_names].values.T.tolist()],
#                                                              l2_y_train, 42, 50, roc_auc_binary_loss_proba)
# l1_ges_train_score = eval_metric(l2_y_train, np.average(l2_train_data[oof_col_names], axis=1, weights=l1_ges_weights))
# l1_ges_test_score = eval_metric(l2_y_test, np.average(l2_test_data[oof_col_names], axis=1, weights=l1_ges_weights))
#
# from autogluon.core.utils.utils import CVSplitter
#
# cv = CVSplitter(n_splits=8, n_repeats=1, stratified=True, random_state=2)
# fold_fake_model_map = {}
# for fold_i, (train_i, test_i) in enumerate(cv.split(l2_X_train, l2_y_train)):
#     fold_fake_model_map[fold_i] = caruana_weighted([np.array(x) for x in l2_train_data.iloc[train_i][oof_col_names].values.T.tolist()],
#                                                              l2_y_train[train_i], 42, 50, roc_auc_binary_loss_proba)[0]
# oof_X = l2_X_train[oof_col_names]
# l1_fold_model_perf = {x: {} for x in oof_col_names}
# for fold in np.unique(l1_fold_indicator):
#     fold_mask = fold == l1_fold_indicator
#     for col in list(oof_col_names):
#         l1_fold_model_perf[col][fold] = [eval_metric(l2_y_train[fold_mask], oof_X.loc[fold_mask, col])]
#
# oof_col_to_rank_per_fold = {
#     oof_col: {k: v[0] for k, v in
#               pd.DataFrame.from_dict(l1_fold_model_perf[oof_col]).rank(axis=1, ascending=False).to_dict(
#                   orient="list").items()}
#     for oof_col in l1_fold_model_perf.keys()
# }
#
# rank_oof_df = oof_X.copy()
# for fold in np.unique(l1_fold_indicator):
#     fold_mask = fold == l1_fold_indicator
#     for col in list(oof_col_names):
#         rank_oof_df[col][fold_mask] = oof_col_to_rank_per_fold[col][fold]
#
# rank_per_instance = rank_oof_df.mean(axis=1)
# avg_rank = rank_per_instance.mean()
# # print('overall avg', avg_rank)
# diff = []
# for fold in np.unique(l2_fold_indicator):
#     fold_mask = fold == l2_fold_indicator
#     print(rank_per_instance[fold_mask].mean())
#     diff.append(abs(avg_rank - rank_per_instance[fold_mask].mean()))
# print('Avg diff in ranks per fold', np.mean(diff), f"(U: {len(np.unique(rank_per_instance))})")
#
# l2_fold_indicator_strat = get_l_fold_indicator(l2_X_train, l2_y_train, layer=2,
#                                                stratify_on=rank_per_instance.astype(str))
#
# rank_per_instance = rank_oof_df.mean(axis=1)
# avg_rank = rank_per_instance.mean()
# diff = []
# for fold in np.unique(l2_fold_indicator_strat):
#     fold_mask = fold == l2_fold_indicator_strat
#     # print(rank_per_instance[fold_mask].mean())
#     diff.append(abs(avg_rank - rank_per_instance[fold_mask].mean()))
# print('Avg diff in ranks per fold', np.mean(diff), f"(U: {len(np.unique(rank_per_instance))})")
#
# # stratify_on = rank_per_instance.astype(str)
# stratify_on = rank_per_instance.astype(str) + "L" + l2_y_train.astype(str)
# l2_fold_indicator_strat = get_l_fold_indicator(l2_X_train, l2_y_train, layer=2,
#                                                stratify_on=rank_per_instance.astype(str) + "L" + l2_y_train.astype(str))
#
# rank_per_instance = rank_oof_df.mean(axis=1)
# avg_rank = rank_per_instance.mean()
# diff = []
# for fold in np.unique(l2_fold_indicator_strat):
#     fold_mask = fold == l2_fold_indicator_strat
#     # print(rank_per_instance[fold_mask].mean())
#     diff.append(abs(avg_rank - rank_per_instance[fold_mask].mean()))
# print('Avg diff in ranks per fold', np.mean(diff), f"(U: {len(np.unique(rank_per_instance))})")


#     print('Duplicates', np.mean(l2_train_data.drop(columns=[label]).duplicated()))
#     mask = ~l2_train_data.drop(columns=[label]).duplicated()
#     l2_train_data = l2_train_data[mask]



# --- to multiclassd stuff
#
# from autogluon.core.metrics import get_metric
# from functools import partial
#
# d_roc_auc = get_metric("roc_auc", problem_type="binary")
#
#
# def _to_binary_proba(y_pred_proba, label_map):
#     sel_proba = [int(k) for k, v in label_map.items() if v == 1]
#     binary_proba = y_pred_proba[:, sel_proba].sum(axis=1)
#     return binary_proba
#
#
# def mc_roc_auc(y_train, y_pred_proba, label_map, post_hoc=False):
#     binary_proba = _to_binary_proba(y_pred_proba, label_map)
#
#     if post_hoc:
#         binary_y_train = y_train
#     else:
#         binary_y_train = y_train.map(label_map)
#
#     return d_roc_auc(binary_y_train, binary_proba)
# def to_multiclass(train_data, label, fit_para, predictor_para, **kwargs):
#     method_name = "to_multiclass"
#     train_data = train_data.copy()
#     from autogluon.core.metrics import make_scorer
#     import pandas as pd
#     # get new labels
#     from sklearn.cluster import KMeans
#     from sklearn.compose import ColumnTransformer, make_column_selector
#     from sklearn.impute import SimpleImputer
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
#
#     preprocessor = Pipeline(
#         [
#             (
#                 "fix",
#                 ColumnTransformer(
#                     transformers=[
#                         (
#                             "num",
#                             SimpleImputer(strategy="constant", fill_value=-1),
#                             make_column_selector(dtype_exclude=["object", "category"]),
#                         ),
#                         (
#                             "cat",
#                             Pipeline(
#                                 steps=[
#                                     (
#                                         "encoder",
#                                         OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
#                                     ),
#                                     ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
#                                 ]
#                             ),
#                             make_column_selector(dtype_include=["object", "category"]),
#                         ),
#                     ],
#                     sparse_threshold=0,
#                 ),
#             ),
#             ("scale", StandardScaler()),
#         ]
#     )
#
#     # _clusterer = KMeans(n_clusters=2)
#     # cluster_i = _clusterer.fit_predict(preprocessor.fit_transform(train_data.drop(columns=[label])))
#     rng = np.random.RandomState(24)
#     # cluster_i = pd.Series(np.full(len(train_data), "-1"))
#
#     sel = rng.choice(len(train_data), len(train_data), replace=False)
#
#     tmp = pd.Series(np.full(len(train_data), ""))
#     tmp[sel[:len(train_data)//2]] = f"_1"
#     tmp[sel[len(train_data)//2:]] = f"_2"
#     cluster_i = tmp
#     print(np.unique(cluster_i))
#
#     # cluster_i[cluster_i == "-1"] = "-1_0_1"
#
#
#     _labels = LabelEncoder().fit_transform(train_data[label])
#
#     new_labels = 'L' + pd.Series(_labels.astype(str))
#     r_mask = new_labels == 'L0'
#     new_labels[r_mask] += pd.Series(cluster_i[r_mask].astype(str))
#     oe = OrdinalEncoder()
#     new_labels = oe.fit_transform(new_labels.values.reshape(-1, 1))
#     new_labels = pd.Series(new_labels[:, 0]).astype(int)
#     print(np.unique(new_labels))
#
#     label_map = {int(row[1]): row[0] for _, row in pd.DataFrame([_labels, new_labels]).T.drop_duplicates().iterrows()}
#
#     # Map new labels to old labels
#     train_data[label] = new_labels
#
#     # def _transform_labels(X, curr_labels):
#     #     f_cluster_i = _clusterer.predict(preprocessor.transform(X))
#     #     f_new_labels = pd.Series(f_cluster_i.astype(str)) + '_' + pd.Series(curr_labels.astype(str))
#     #     # there might be new labels created here, need to fix this
#     #     f_new_labels = oe.transform(f_new_labels.values.reshape(-1, 1))
#     #     f_new_labels = pd.Series(f_new_labels[:, 0]).astype(int)
#     #     return f_new_labels
#
#     ag_roc_auc_scorer = make_scorer(name='roc_auc',
#                                     score_func=partial(mc_roc_auc,
#                                                        label_map=label_map),
#                                     optimum=1,
#                                     greater_is_better=True,
#                                     needs_proba=True)
#
#     logger.debug(f"Start running AutoGluon on data: {method_name}")
#     predictor_para = predictor_para.copy()
#     predictor_para["problem_type"] = "multiclass"
#     predictor_para["eval_metric"] = ag_roc_auc_scorer
#     predictor_para["learner_kwargs"]["label_count_threshold"] = 0
#     predictor = TabularPredictor(**predictor_para)
#     predictor.fit(train_data=train_data, **fit_para)
#     predictor.post_hoc_metric = make_scorer(name='roc_auc',
#                                     score_func=partial(mc_roc_auc,
#                                                        label_map=label_map, post_hoc=True),
#                                     optimum=1,
#                                     greater_is_better=True,
#                                     needs_proba=True)
#     predictor._to_binary_proba = partial(_to_binary_proba, label_map=label_map)
#
#     return predictor, method_name, None
#
