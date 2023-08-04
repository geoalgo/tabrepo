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