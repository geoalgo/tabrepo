import numpy as np
import pandas as pd


def _basic_flip(self, X, y_pred_proba, val_label):
    l1_preds = X[self.params_aux['l1_model_worst_to_best']].values
    l2_loss = abs(val_label - y_pred_proba)
    l1_loss = np.mean(abs(l1_preds - val_label.values.reshape(-1, 1)), axis=1)

    threshold = 0.15
    flipped_mask = (l1_loss - l2_loss) >= threshold
    y_pred_proba[flipped_mask] = np.mean(l1_preds, axis=1)[flipped_mask]


def _wo_label_info_flip(self, X, y_pred_proba):
    stack_features = self.feature_metadata.get_features(required_special_types=['stack'])

    # -- Relative v3 (not-label based)
    # does not work, needs to be relative to label or something?
    one_threshold = 0.25
    l1_pred = np.average(X[stack_features].values, axis=1, weights=self.params_aux['l1_ges_weights'])
    gap = abs(l1_pred - y_pred_proba)
    flipped_mask = (gap >= one_threshold)
    y_pred_proba[flipped_mask] = l1_pred[flipped_mask]


def _dynamic_threshold_v1(self, X, y_pred_proba, val_label):
    from scipy.stats import iqr

    l1_loss_all = abs(X[self.params_aux['l1_model_worst_to_best']].values - val_label.values.reshape(-1, 1))

    # -- Get threshold
    proxy_l1_loss = l1_loss_all[:, 0]  # worst l1 model
    proxy_l2_loss = l1_loss_all[:, -1]  # best l2 model

    # - Threshold rel l1: val becomes worse
    g = (proxy_l1_loss - proxy_l2_loss) / proxy_l1_loss
    g = g[proxy_l2_loss < proxy_l1_loss]

    loss_closed_gap_threshold_relative_to_l1 = np.max(g[g <= (np.quantile(g, 0.75) + 1.5 * iqr(g))]) if len(
        g) > 0 else 1

    # - Threshold rel l2: val becomes better
    g = (proxy_l2_loss - proxy_l1_loss) / proxy_l2_loss
    g = g[proxy_l2_loss > proxy_l1_loss]
    loss_closed_gap_threshold_relative_to_l2 = np.max(g[g <= (np.quantile(g, 0.75) + 1.5 * iqr(g))]) if len(
        g) > 0 else 1


def _flip_ranks_not_proba_try(self, X, y_pred_proba, val_label):
    from scipy.stats import rankdata

    stack_features = self.feature_metadata.get_features(required_special_types=['stack'])

    l2_loss = abs(val_label - y_pred_proba)
    l1_loss_all = abs(X[self.params_aux['l1_model_worst_to_best']].values - val_label.values.reshape(-1, 1))

    rank_map = {k: v for v, k in zip(y_pred_proba, rankdata(y_pred_proba, method='ordinal'))}

    def v(x):
        return rank_map[x]

    v_func = np.vectorize(v)

    # -- Relative v2 (static threshold)
    loss_closed_gap_threshold_relative_to_l1 = 0.5  # val becomes worse
    loss_closed_gap_threshold_relative_to_l2 = 0.5  # val becomes better

    # flip more than threshold% decrease in loss (make val worse)
    l1_loss = np.max(l1_loss_all, axis=1)
    gap_closed = (l1_loss - l2_loss) / l1_loss
    flipped_mask = (gap_closed >= loss_closed_gap_threshold_relative_to_l1) & (l2_loss < l1_loss)
    new_ranks = rankdata(X[stack_features].values[np.arange(len(X)), np.argmax(l1_loss_all, axis=1)],
                         method='ordinal')[flipped_mask]
    y_pred_proba[flipped_mask] = v_func(new_ranks)

    # flip more than threshold% increase in loss (make val better)
    l1_loss = np.min(l1_loss_all, axis=1)
    gap_closed = (l2_loss - l1_loss) / l2_loss
    flipped_mask = (gap_closed >= loss_closed_gap_threshold_relative_to_l2) & (l2_loss > l1_loss)
    new_ranks = rankdata(X[stack_features].values[np.arange(len(X)), np.argmin(l1_loss_all, axis=1)],
                         method='ordinal')[flipped_mask]
    y_pred_proba[flipped_mask] = v_func(new_ranks)


def val_flip(self, X, y_pred_proba, val_label):
    stack_features = self.feature_metadata.get_features(required_special_types=['stack'])

    l2_loss = abs(val_label - y_pred_proba)
    l1_loss_all = abs(X[self.params_aux['l1_model_worst_to_best']].values - val_label.values.reshape(-1, 1))

    # -- Relative v2 (static threshold)
    loss_closed_gap_threshold_relative_to_l1 = 0.5  # val becomes worse
    loss_closed_gap_threshold_relative_to_l2 = 0.5  # val becomes better

    # flip more than threshold% decrease in loss (make val worse)
    l1_loss = np.max(l1_loss_all, axis=1)
    gap_closed = (l1_loss - l2_loss) / l1_loss
    flipped_mask = (gap_closed >= loss_closed_gap_threshold_relative_to_l1) & (l2_loss < l1_loss)
    y_pred_proba[flipped_mask] = \
        X[stack_features].values[np.arange(len(X)), np.argmax(l1_loss_all, axis=1)][flipped_mask]

    # flip more than threshold% increase in loss (make val better)
    l1_loss = np.min(l1_loss_all, axis=1)
    gap_closed = (l2_loss - l1_loss) / l2_loss
    flipped_mask = (gap_closed >= loss_closed_gap_threshold_relative_to_l2) & (l2_loss > l1_loss)
    y_pred_proba[flipped_mask] = \
        X[stack_features].values[np.arange(len(X)), np.argmin(l1_loss_all, axis=1)][flipped_mask]


def non_loss_based_flip_best(self, X, y_pred_proba, val_label):
    # --- Required during fit:
    # stack_f = self.feature_metadata.get_features(required_special_types=['stack'])
    #
    # from scripts.leakage_benchmark.src.other.post_hoc_ensembling import caruana_weighted, \
    #     roc_auc_binary_loss_proba
    # self._l1_ges_weights, _ = caruana_weighted([np.array(x) for x in X[stack_f].values.T.tolist()],
    #                                            y, 42, 50, roc_auc_binary_loss_proba)
    # self._stack_f = stack_f

    reasonable_l2_proba = np.average(X[self._stack_f], axis=1, weights=self._l1_ges_weights)
    diff = reasonable_l2_proba - y_pred_proba

    if val_label is not None:
        # -- Find threshold
        l2_loss = abs(val_label - y_pred_proba)
        reasonable_l2_loss = abs(val_label - reasonable_l2_proba)
        rel_loss_diff = (reasonable_l2_loss - l2_loss) / np.max([reasonable_l2_loss, l2_loss], axis=0)

        # Loss reduced flip (cheating mapping)
        allowed_gap = 0.3  # ?gap allowed between a linear l2 model and a non-linear l2 model?
        win_cheat_cases = diff[rel_loss_diff >= allowed_gap]
        self._win_cheat_th_greater = np.min(win_cheat_cases[win_cheat_cases > 0]) if sum(win_cheat_cases > 0) else None
        self._win_cheat_th_smaller = np.max(win_cheat_cases[win_cheat_cases < 0]) if sum(
            win_cheat_cases < 0) else None

        # Loss increased flip (failure due to cheating mapping)
        lose_cheat_cases = diff[rel_loss_diff <= -allowed_gap]
        self._lose_cheat_th_greater = np.min(lose_cheat_cases[lose_cheat_cases > 0]) if sum(
            lose_cheat_cases > 0) else None
        self._lose_cheat_th_smaller = np.max(lose_cheat_cases[lose_cheat_cases < 0]) if sum(
            lose_cheat_cases < 0) else None

    # -- Flip based on learned smaller/greater mapping

    # TODO: think about this impact that they overlap
    #   - can just take abs above and flip once here.
    #   - found no way to mirror the loss relationship down here

    # Flip to make model worse
    if self._win_cheat_th_greater is not None:
        mask_flip_to_worse_greater = diff >= self._win_cheat_th_greater
        y_pred_proba[mask_flip_to_worse_greater] = reasonable_l2_proba[mask_flip_to_worse_greater]

    if self._win_cheat_th_smaller is not None:
        mask_flip_to_worse_smaller = diff <= self._win_cheat_th_smaller
        y_pred_proba[mask_flip_to_worse_smaller] = reasonable_l2_proba[mask_flip_to_worse_smaller]

    # Flip to make model better
    if self._lose_cheat_th_greater is not None:
        mask_flip_to_better_greater = diff >= self._lose_cheat_th_greater
        y_pred_proba[mask_flip_to_better_greater] = reasonable_l2_proba[mask_flip_to_better_greater]

    if self._lose_cheat_th_smaller is not None:
        mask_flip_to_better_smaller = diff <= self._lose_cheat_th_smaller
        y_pred_proba[mask_flip_to_better_smaller] = reasonable_l2_proba[mask_flip_to_better_smaller]


def _with_norm(self, X, y_pred_proba, val_label):
    from sklearn.preprocessing import MinMaxScaler

    # org_y_pred_proba = y_pred_proba.copy()
    reasonable_l2_proba = np.average(X[self._stack_f], axis=1, weights=self._l1_ges_weights)
    allowed_gap = 0.3

    def _get_masks(labels, gap, scale_gap=True):
        l2_loss = abs(labels - y_pred_proba)
        reasonable_l2_loss = abs(labels - reasonable_l2_proba)
        rel_loss_diff = (reasonable_l2_loss - l2_loss) / np.max([reasonable_l2_loss, l2_loss], axis=0)

        if scale_gap:
            # Scale gap measure by observed impact of loss change
            # higher is more impact
            impact_of_change = abs(reasonable_l2_loss - l2_loss) / np.mean(abs(reasonable_l2_loss - l2_loss))
            # make needed gap smaller where impact is larger
            _gap = np.full_like(rel_loss_diff, gap) / (impact_of_change + np.finfo(float).eps)
        else:
            _gap = gap

        _win_cheat_cases = rel_loss_diff >= _gap
        _lose_cheat_cases = rel_loss_diff <= -_gap
        return _win_cheat_cases, _lose_cheat_cases

    def _transform(_l2_scaler, _reasonable_l2_scaler, _clip_min, _clip_max, x):
        return _l2_scaler.inverse_transform(
            _reasonable_l2_scaler.transform(np.clip(x, _clip_min, _clip_max).reshape(-1, 1))
        ).flatten()

    if val_label is not None:
        win_cheat_cases, lose_cheat_cases = _get_masks(val_label, allowed_gap)

        tmp = y_pred_proba.copy()
        _u, _l = min(tmp.mean() + tmp.std() * 3, 1), max(tmp.mean() - tmp.std() * 3, 0)
        tmp = np.append(tmp, _u)
        tmp = np.append(tmp, _l)

        tmp_r = reasonable_l2_proba.copy()
        _u, _l = min(tmp_r.mean() + tmp_r.std() * 3, 1), max(tmp_r.mean() - tmp_r.std() * 3, 0)
        tmp_r = np.append(tmp_r, _u)
        tmp_r = np.append(tmp_r, _l)

        self._l2_scaler = MinMaxScaler().fit(tmp.reshape(-1, 1))
        self._reasonable_l2_scaler = MinMaxScaler().fit(tmp_r.reshape(-1, 1))
        self._clip_min = min(tmp_r)
        self._clip_max = max(tmp_r)
        print(abs(min(tmp_r) - min(tmp)), abs(max(tmp_r) - max(tmp)))
    else:
        pseudo_label = (reasonable_l2_proba >= 0.5).astype(int)
        win_cheat_cases, lose_cheat_cases = _get_masks(pseudo_label, allowed_gap)

    # -- Flip based on learned smaller/greater mapping

    # Flip to make model worse
    if sum(win_cheat_cases):
        y_pred_proba[win_cheat_cases] = _transform(self._l2_scaler, self._reasonable_l2_scaler, self._clip_min,
                                                   self._clip_max,
                                                   reasonable_l2_proba[win_cheat_cases])

    # Flip to make model better
    if sum(lose_cheat_cases):
        y_pred_proba[lose_cheat_cases] = _transform(self._l2_scaler, self._reasonable_l2_scaler, self._clip_min,
                                                    self._clip_max,
                                                    reasonable_l2_proba[lose_cheat_cases])


def current_best(self, X, y_pred_proba, val_label):
    # --- Required during fit:
    # stack_f = self.feature_metadata.get_features(required_special_types=['stack'])
    #
    # from scripts.leakage_benchmark.src.other.post_hoc_ensembling import caruana_weighted, \
    #     roc_auc_binary_loss_proba
    # self._l1_ges_weights, _ = caruana_weighted([np.array(x) for x in X[stack_f].values.T.tolist()],
    #                                            y, 42, 50, roc_auc_binary_loss_proba)
    # self._stack_f = stack_f


    stack_f_col_indices = [i for i, f_name in enumerate(self.features) if f_name in self._stack_f]
    if isinstance(X, pd.DataFrame):
        reasonable_l2_proba = np.average(X.values[:, stack_f_col_indices], axis=1, weights=self._l1_ges_weights)
    else:
        reasonable_l2_proba = np.average(X[:, stack_f_col_indices], axis=1, weights=self._l1_ges_weights)
    allowed_gap = 0.3

    def _get_masks(labels, gap, scale_gap=True):
        l2_loss = abs(labels - y_pred_proba)
        reasonable_l2_loss = abs(labels - reasonable_l2_proba)
        rel_loss_diff = (reasonable_l2_loss - l2_loss) / np.max([reasonable_l2_loss, l2_loss], axis=0)

        if scale_gap:
            # Scale gap measure by observed impact of loss change
            # higher is more impact
            impact_of_change = abs(reasonable_l2_loss - l2_loss) / np.mean(abs(reasonable_l2_loss - l2_loss))
            # make needed gap smaller where impact is larger
            _gap = np.full_like(rel_loss_diff, gap) / (impact_of_change + np.finfo(float).eps)
        else:
            _gap = gap

        _win_cheat_cases = rel_loss_diff >= _gap
        _lose_cheat_cases = rel_loss_diff <= -_gap
        return _win_cheat_cases, _lose_cheat_cases

    if val_label is not None:
        win_cheat_cases, lose_cheat_cases = _get_masks(val_label, allowed_gap)
    else:
        pseudo_label = (reasonable_l2_proba >= 0.5).astype(int)
        win_cheat_cases, lose_cheat_cases = _get_masks(pseudo_label, allowed_gap)

    # -- Flip based on learned smaller/greater mapping
    # Flip to make model worse
    y_pred_proba[win_cheat_cases] = reasonable_l2_proba[win_cheat_cases]
    # Flip to make model better
    y_pred_proba[lose_cheat_cases] = reasonable_l2_proba[lose_cheat_cases]


def tmp():
    def _apply_leak_protection(
        self,
        X: pd.DataFrame | np.ndarray,
        y_pred_proba: np.ndarray,
        y_val_fold: np.ndarray | None,
        allowed_gap: float = 0.3,
        scale_gap: bool = True,
        fit_single_proba: np.ndarray = None,
    ):
        """
        Adjust the predictions to avoid stack info leakage.
        FIXME: likely we want to increase the gap for multi-class as definition is a bit different...
        """

        # Technically, we would need to preprocess X here. But as we are only interested in
        # the stack features, we ignore this for now.
        if fit_single_proba is not None:
            # Edge case for models that do not use cross-validation (e.g., RF that uses OOB)
            reasonable_proba = fit_single_proba
        else:
            reasonable_proba = self._reasonable_ensemble.predict_proba(X)

        def _get_masks(labels, gap):
            if self.problem_type == MULTICLASS:
                # FIXME: need to check that len(unique(y_val_fold)) == proba.shape[1], does ag guarantee this?
                from sklearn.preprocessing import \
                    LabelBinarizer  # fixme, move up or keep lazy?

                lb = LabelBinarizer().fit(labels)
                lb.classes_ = list(range(y_pred_proba.shape[-1]))
                transformed_labels = lb.transform(labels)

                loss = 1 - (transformed_labels * y_pred_proba).sum(axis=1)
                reasonable_loss = 1 - (transformed_labels * reasonable_proba).sum(axis=1)
            elif self.problem_type in [BINARY, REGRESSION]:
                # FIXME: assumption that labels are 0 and 1 for binary
                loss = abs(labels - y_pred_proba)
                reasonable_loss = abs(labels - reasonable_proba)
            else:
                # TODO: catch this earlier during init
                raise ValueError(f"Unknown problem type for leak protection: {self.problem_type}")

            # FIXME: Might need to add esp here too or handle 0 loss cases differently
            #   Currently, they are simply flipped as it makes no difference if they are the same.
            rel_loss_diff = (reasonable_loss - loss) / np.max([reasonable_loss, loss], axis=0)

            if scale_gap:
                # Scale gap measure by observed impact of loss change
                # higher is more impact
                impact_of_change = abs(reasonable_loss - loss) / np.mean(abs(reasonable_loss - loss))
                # make needed gap smaller where impact is larger
                _gap = np.full_like(rel_loss_diff, gap) / (impact_of_change + np.finfo(float).eps)
            else:
                _gap = gap

            _win_cheat_cases = rel_loss_diff >= _gap
            _lose_cheat_cases = rel_loss_diff <= -_gap
            return _win_cheat_cases, _lose_cheat_cases

        if y_val_fold is not None and False: # and False:
            from sklearn.preprocessing import MinMaxScaler

            win_cheat_cases, lose_cheat_cases = _get_masks(y_val_fold, allowed_gap)

            tmp = y_pred_proba.copy()
            _u, _l = min(tmp.mean() + tmp.std() * 3, 1), max(tmp.mean() - tmp.std() * 3, 0)
            tmp = np.append(tmp, _u)
            tmp = np.append(tmp, _l)

            tmp_r = reasonable_proba.copy()
            _u, _l = min(tmp_r.mean() + tmp_r.std() * 3, 1), max(tmp_r.mean() - tmp_r.std() * 3, 0)
            tmp_r = np.append(tmp_r, _u)
            tmp_r = np.append(tmp_r, _l)

            self._l2_scaler = MinMaxScaler().fit(tmp.reshape(-1, 1))
            self._reasonable_l2_scaler = MinMaxScaler().fit(tmp_r.reshape(-1, 1))
            self._clip_min = min(tmp_r)
            self._clip_max = max(tmp_r)
        else:
            if self.problem_type == MULTICLASS:
                # FIXME: assumption that order of proba vectors respects order of classes and labels are 0...N.
                pseudo_label = reasonable_proba.argmax(axis=1)
            elif self.problem_type == BINARY:
                # FIXME: assumption that the optimal threshold is 0.5
                pseudo_label = (reasonable_proba >= 0.5).astype(int)
            else:
                # FIXME: this idea won't work well for regression due to no difference here and the impact on loss
                #   -> will always flip unless gap is >= 100% due to impact scaler
                #   -> will only work for validation data and in extreme abs loss difference cases... mhm
                pseudo_label = reasonable_proba

            win_cheat_cases, lose_cheat_cases = _get_masks(pseudo_label, allowed_gap)

        # -- Flip based on masks
        # Flip to make model worse
        y_pred_proba[win_cheat_cases] = reasonable_proba[win_cheat_cases]
        # Flip to make model better
        y_pred_proba[lose_cheat_cases] = reasonable_proba[lose_cheat_cases]

        # def _transform(_l2_scaler, _reasonable_l2_scaler, _clip_min, _clip_max, x):
        #     return _l2_scaler.inverse_transform(
        #         _reasonable_l2_scaler.transform(np.clip(x, _clip_min, _clip_max).reshape(-1, 1))
        #     ).flatten()
        #
        # if sum(win_cheat_cases):
        #     y_pred_proba[win_cheat_cases] = _transform(self._l2_scaler, self._reasonable_l2_scaler, self._clip_min,
        #                                                self._clip_max,
        #                                                reasonable_proba[win_cheat_cases])
        #
        # # Flip to make model better
        # if sum(lose_cheat_cases):
        #     y_pred_proba[lose_cheat_cases] = _transform(self._l2_scaler, self._reasonable_l2_scaler, self._clip_min,
        #                                                 self._clip_max,
        #                                                 reasonable_proba[lose_cheat_cases])

        return y_pred_proba



def v2_bench():
    if fit_single_proba is not None:
        # Edge case for models that do not use cross-validation (e.g., RF that uses OOB)
        reasonable_proba = fit_single_proba
    else:
        reasonable_proba = self._reasonable_ensemble.predict_proba(X)

    def _get_masks(labels):
        _gap = allowed_gap
        if self.problem_type == MULTICLASS:
            # FIXME: need to check that len(unique(y_val_fold)) == proba.shape[1], does ag guarantee this?
            from sklearn.preprocessing import \
                LabelBinarizer  # fixme, move up or keep lazy?
            _gap *= 2
            lb = LabelBinarizer().fit(labels)
            lb.classes_ = list(range(y_pred_proba.shape[-1]))
            transformed_labels = lb.transform(labels)

            loss = 1 - (transformed_labels * y_pred_proba).sum(axis=1)
            reasonable_loss = 1 - (transformed_labels * reasonable_proba).sum(axis=1)
        elif self.problem_type in [BINARY, REGRESSION]:
            # FIXME: assumption that labels are 0 and 1 for binary
            loss = abs(labels - y_pred_proba)
            reasonable_loss = abs(labels - reasonable_proba)
        else:
            # TODO: catch this earlier during init
            raise ValueError(f"Unknown problem type for leak protection: {self.problem_type}")

        # -- Identical to the last version, just reformulated
        abs_diff = abs(reasonable_loss - loss)
        # trim mean is the inter-quartile mean
        all_cases = abs_diff >= np.full_like(reasonable_loss, _gap) * np.max([reasonable_loss, loss], axis=0)
        import scipy.stats as stat

        ##print(max(abs_diff), min(abs_diff), np.median(abs_diff),
        #     trim_mean(abs_diff, 0.25), np.mean(abs_diff), stat.gmean(abs_diff), np.quantile(abs_diff, 0.25), np.quantile(abs_diff, 0.75), np.quantile(abs_diff, 0.9),
        #    stat.iqr(abs_diff))
        print(sum(all_cases) / len(all_cases))
        return all_cases

    if self._flip_normalization:
        from sklearn.preprocessing import MinMaxScaler
        tmp = y_pred_proba.copy()
        if self.problem_type == BINARY:
            _u, _l = min(tmp.mean() + tmp.std() * 3, 1), max(tmp.mean() - tmp.std() * 3, 0)
        else:
            _u, _l = tmp.mean() + tmp.std() * 3, tmp.mean() - tmp.std() * 3

        tmp = np.append(tmp, _u)
        tmp = np.append(tmp, _l)

        tmp_r = reasonable_proba.copy()
        if self.problem_type == BINARY:
            _u, _l = min(tmp_r.mean() + tmp_r.std() * 3, 1), max(tmp_r.mean() - tmp_r.std() * 3, 0)
        else:
            _u, _l = tmp.mean() + tmp.std() * 3, tmp.mean() - tmp.std() * 3
        tmp_r = np.append(tmp_r, _u)
        tmp_r = np.append(tmp_r, _l)

        self._l2_scaler = MinMaxScaler().fit(tmp.reshape(-1, 1))
        self._reasonable_l2_scaler = MinMaxScaler().fit(tmp_r.reshape(-1, 1))
        self._clip_min = min(tmp_r)
        self._clip_max = max(tmp_r)

    if self.problem_type == MULTICLASS:
        # FIXME: assumption that order of proba vectors respects order of classes and labels are 0...N.
        pseudo_label = reasonable_proba.argmax(axis=1)
    elif self.problem_type == BINARY:
        # FIXME: assumption that the optimal threshold is 0.5
        pseudo_label = (reasonable_proba >= 0.5).astype(int)
    else:
        # FIXME: this idea won't work well for regression due to no difference here and the impact on loss
        #   -> will always flip unless gap is >= 100% due to impact scaler
        #   -> will only work for validation data and in extreme abs loss difference cases... mhm
        pseudo_label = reasonable_proba

    all_cases = _get_masks(pseudo_label)

    # -- Flip based on masks
    if self._flip_normalization:
        def _transform(_l2_scaler, _reasonable_l2_scaler, _clip_min, _clip_max, x):
            return _l2_scaler.inverse_transform(
                _reasonable_l2_scaler.transform(np.clip(x, _clip_min, _clip_max).reshape(-1, 1))
            ).flatten()

        if sum(all_cases):
            y_pred_proba[all_cases] = _transform(self._l2_scaler, self._reasonable_l2_scaler, self._clip_min,
                                                 self._clip_max, reasonable_proba[all_cases])
    else:
        y_pred_proba[all_cases] = reasonable_proba[all_cases]

    return y_pred_proba


def _failed_cluster_test():
    def _apply_leak_protection(
            self,
            X: pd.DataFrame | np.ndarray,
            y_pred_proba: np.ndarray,
            y_val_fold: np.ndarray | None,
            allowed_gap: float = 0.5,
            fit_single_proba: np.ndarray = None,
    ):
        """
        Adjust the predictions to avoid stack info leakage.
        FIXME: likely we want to increase the gap for multi-class as definition is a bit different...
        """
        from scipy.stats import trim_mean

        # Technically, we would need to preprocess X here. But as we are only interested in
        # the stack features, we ignore this for now.
        if fit_single_proba is not None:
            # Edge case for models that do not use cross-validation (e.g., RF that uses OOB)
            reasonable_proba = fit_single_proba
        else:
            reasonable_proba = self._reasonable_ensemble.predict_proba(X)


        if y_val_fold is not None:
            from functools import partial

            from sklearn.cluster import KMeans
            self._clusterer = KMeans(n_clusters=1)
            self._cluster_pre = self._preprocessor_for_cluster()
            cluster_i = self._clusterer.fit_predict(self._cluster_pre.fit_transform(X))
            self._cluster_map = {}

            for i in np.unique(cluster_i):
                cluster_mask = cluster_i == i
                _m = partial(trim_mean, proportiontocut=0.25) if sum(cluster_mask) > 10 else np.mean
                ref = _m(reasonable_proba[cluster_mask])
                self._cluster_map[i] = [
                    ref,
                    _m(np.abs((abs(ref - y_pred_proba[cluster_mask]) - abs(ref - reasonable_proba[cluster_mask])))),
                ]
        else:
            cluster_i = self._clusterer.predict(self._cluster_pre.transform(X))

        flip_mask = np.full_like(cluster_i, False)
        for i, (point_of_reference, impactor) in self._cluster_map.items():
            cluster_mask = cluster_i == i
            if not sum(cluster_mask):
                continue
            y_cluster = y_pred_proba[cluster_mask]
            r_y_cluster = reasonable_proba[cluster_mask]

            l2_loss = np.abs(point_of_reference - y_cluster)
            reasonable_l2_loss = np.abs(point_of_reference - r_y_cluster)

            rel_loss_diff = np.abs(l2_loss - reasonable_l2_loss) / (l2_loss + reasonable_l2_loss)

            impact_of_change = np.abs(reasonable_l2_loss - l2_loss) / impactor

            flip_cases = rel_loss_diff >= (np.full_like(rel_loss_diff, allowed_gap) / (impact_of_change + np.finfo(float).eps))
            flip_indices = np.where(cluster_mask)[0][flip_cases]

            flip_mask[flip_indices] = True
        print(np.mean(flip_mask))
        y_pred_proba[flip_mask] = reasonable_proba[flip_mask]
        return y_pred_proba