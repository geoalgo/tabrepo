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