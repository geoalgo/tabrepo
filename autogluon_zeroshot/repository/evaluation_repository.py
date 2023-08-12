from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from autogluon_zeroshot.simulation.configuration_list_scorer import ConfigurationListScorer
from autogluon_zeroshot.simulation.ensemble_selection_config_scorer import EnsembleSelectionConfigScorer
from autogluon_zeroshot.simulation.simulation_context import ZeroshotSimulatorContext
from autogluon_zeroshot.simulation.single_best_config_scorer import SingleBestConfigScorer
from autogluon_zeroshot.simulation.tabular_predictions import TabularModelPredictions
from autogluon_zeroshot.utils.cache import SaveLoadMixin
from autogluon_zeroshot.utils import catchtime
from autogluon_zeroshot import repository

from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.common.features.feature_metadata import FeatureMetadata


class EvaluationRepository(SaveLoadMixin):
    """
    Simple Repository class that implements core functionality related to
    fetching model predictions, available datasets, folds, etc.
    """

    def __init__(
            self,
            zeroshot_context: ZeroshotSimulatorContext,
            tabular_predictions: TabularModelPredictions,
            ground_truth: dict,
    ):
        self._tabular_predictions: TabularModelPredictions = tabular_predictions
        self._zeroshot_context: ZeroshotSimulatorContext = zeroshot_context
        self._ground_truth: dict = ground_truth
        assert all(x in self._tid_to_name for x in self._tabular_predictions.datasets)

    def to_zeroshot(self) -> repository.EvaluationRepositoryZeroshot:
        """
        Returns a version of the repository as an EvaluationRepositoryZeroshot object.

        :return: EvaluationRepositoryZeroshot object
        """
        from autogluon_zeroshot.repository import EvaluationRepositoryZeroshot
        self_zeroshot = copy.copy(self)  # Shallow copy so that the class update does not alter self
        self_zeroshot.__class__ = EvaluationRepositoryZeroshot
        return self_zeroshot

    def print_info(self):
        self._zeroshot_context.print_info()

    @property
    def _name_to_tid(self) -> Dict[str, int]:
        return self._zeroshot_context.dataset_to_tid_dict

    @property
    def _tid_to_name(self) -> Dict[int, str]:
        return {v: k for k, v in self._name_to_tid.items()}

    def subset(self,
               folds: List[int] = None,
               models: List[str] = None,
               tids: List[Union[str, int]] = None,
               problem_types: List[str] = None,
               verbose: bool = True,
               ):
        """
        Method to subset the repository object and force to a dense representation.

        :param folds: The list of folds to subset. Ignored if unspecified.
        :param models: The list of models to subset. Ignored if unspecified.
        :param tids: The list of dataset task ids to subset. Ignored if unspecified.
        :param problem_types: The list of problem types to subset. Ignored if unspecified.
        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return self after in-place updates in this call.
        """
        if folds:
            self._zeroshot_context.subset_folds(folds=folds)
        if models:
            self._zeroshot_context.subset_models(models=models)
        if tids:
            # TODO: Align `_zeroshot_context` naming of datasets -> tids
            self._zeroshot_context.subset_datasets(datasets=tids)
        if problem_types:
            self._zeroshot_context.subset_problem_types(problem_types=problem_types)
        self.force_to_dense(verbose=verbose)
        return self

    # TODO: Add `is_dense` method to assist in unit tests + sanity checks
    def force_to_dense(self, verbose: bool = True):
        """
        Method to force the repository to a dense representation inplace.

        This will ensure that all datasets contain the same folds, and all tasks contain the same models.
        Calling this method when already in a dense representation will result in no changes.

        :param verbose: Whether to log verbose details about the force to dense operation.
        :return: Return self after in-place updates in this call.
        """
        # TODO: Move these util functions to simulations or somewhere else to avoid circular imports
        from autogluon_zeroshot.contexts.utils import intersect_folds_and_datasets, force_to_dense, prune_zeroshot_gt
        # keep only dataset whose folds are all present
        intersect_folds_and_datasets(self._zeroshot_context, self._tabular_predictions, self._ground_truth)
        force_to_dense(self._tabular_predictions,
                       first_prune_method='task',
                       second_prune_method='dataset',
                       verbose=verbose)

        self._zeroshot_context.subset_models(self._tabular_predictions.models)
        self._zeroshot_context.subset_datasets(self._tabular_predictions.datasets)
        self._tabular_predictions.restrict_models(self._zeroshot_context.get_configs())
        self._ground_truth = prune_zeroshot_gt(zeroshot_pred_proba=self._tabular_predictions,
                                               zeroshot_gt=self._ground_truth,
                                               verbose=verbose)
        return self

    @property
    def _df_metadata(self) -> pd.DataFrame:
        return self._zeroshot_context.df_metadata

    def tids(self, problem_type: str = None) -> List[int]:
        """
        Note: returns the taskid of the datasets rather than the string name.

        :param problem_type: If specified, only datasets with the given problem_type are returned.
        """
        return self._zeroshot_context.get_datasets(problem_type=problem_type)

    def dataset_names(self) -> List[str]:
        tids = self.tids()
        dataset_names = [self._tid_to_name[tid] for tid in tids]
        return dataset_names

    def list_models_available(self, tid: int) -> List[str]:
        # TODO rename with new name, and keep naming convention of tabular_predictions to allow filtering over folds,
        #  datasets, specify whether all need to be present etc
        """
        :param tid:
        :return: the list of configurations that are available on all folds of the given dataset.
        """
        res = set(self._tabular_predictions.list_models_available(datasets=[tid]))
        for fold in self.folds:
            df = self._zeroshot_context.df_results_by_dataset_vs_automl
            task = self.task_name(tid=tid, fold=fold)
            methods = set(df.loc[df.dataset == task, "framework"].unique())
            res = res.intersection(methods)
        return list(sorted(res))

    # TODO: Unify with `list_models_available`
    def list_models(self) -> List[str]:
        """
        List all models that appear in at least one task.

        :return: the list of configurations that are available in at least one task.
        """
        return self._zeroshot_context.get_configs()

    def dataset_to_tid(self, dataset_name: str) -> int:
        return self._name_to_tid[dataset_name]

    def tid_to_dataset(self, tid: int) -> str:
        return self._tid_to_name.get(tid, "Not found")

    def eval_metrics(self, tid: int, config_names: List[str], fold: int, check_all_found: bool = True) -> List[dict]:
        """
        :param tid:
        :param config_names: list of configs to query metrics
        :param fold:
        :return: list of metrics for each configuration
        """
        df = self._zeroshot_context.df_results_by_dataset_vs_automl
        task = self.task_name(tid=tid, fold=fold)
        mask = (df.dataset == task) & (df.framework.isin(config_names))
        output_cols = ["framework", "time_train_s", "metric_error", "time_infer_s", "bestdiff", "loss_rescaled",
                       "time_train_s_rescaled", "time_infer_s_rescaled", "rank", "score_val"]
        if check_all_found:
            assert sum(mask) == len(config_names), \
                f"expected one evaluation occurence for each configuration {config_names} for {tid}, " \
                f"{fold} but found {sum(mask)}."
        return [dict(zip(output_cols, row)) for row in df.loc[mask, output_cols].values]

    def val_predictions(self, tid: int, config_name: str, fold: int) -> np.array:
        val_predictions, _ = self._tabular_predictions.predict(
            dataset=tid,
            fold=fold,
            models=[config_name]
        )
        return val_predictions[0]

    def test_predictions(self, tid: int, config_name: str, fold: int) -> np.array:
        _, test_predictions = self._tabular_predictions.predict(
            dataset=tid,
            fold=fold,
            models=[config_name]
        )
        return test_predictions[0]

    def dataset_metadata(self, tid: int) -> dict:
        metadata = self._df_metadata[self._df_metadata.tid == tid]
        return dict(zip(metadata.columns, metadata.values[0]))

    @staticmethod
    def get_data(tid: int, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get Data from OpenML following the AutoML Benchmark.

        Returns raw data associated to the OpenML task with `tid` whereby the raw data is split
        according to the `fold` index.

        :param tid: task id of a task on OpenML
        :param fold: fold index of a fold associated to the OpenML task.
        :return: train_data, test_data
        """
        # Delayed import for now as this is the only openml usage in this file
        import openml

        # Get Task and dataset from OpenML and return split data
        oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True,
                                         download_qualities=False, download_features_meta_data=False)

        train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
        X, *_ = oml_task.get_dataset().get_data(dataset_format='dataframe')

        return X.iloc[train_ind, :], X.iloc[test_ind, :]

    def preprocess_data(self, tid: int, fold: int, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        reset_index: bool = False) \
            -> [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, FeatureMetadata]:
        """
        Preprocesses the given data for the given task and fold.

        The preprocessing follows the default preprocessing of AutoGluon.

        :param tid: task id of a task on OpenML
        :param fold: fold index of a fold associated to the OpenML task.
        :param train_data: the train data for the tid and fold
        :param test_data: the test data for the tid and fold
        :param reset_index: if True, reset index such that both data subset have an index starting from 0.
        :return: X_train, y_train, X_test, y_test, feature_metadata
        """
        task_ground_truth_metadata: dict = self._ground_truth[tid][fold]
        label = task_ground_truth_metadata['label']

        # Verify repo and data correct split
        y_train = task_ground_truth_metadata['y_val']  # technically y_val but for the sake of this y_train
        y_test = task_ground_truth_metadata['y_test']
        assert bool(np.setdiff1d(y_train.index.to_numpy(), train_data.index.to_numpy())) is False
        assert bool(np.setdiff1d(y_test.index.to_numpy(), test_data.index.to_numpy())) is False

        # Preprocess like AutoGluon
        preprocessor = AutoMLPipelineFeatureGenerator()
        X_train, y_train = train_data.drop(labels=[label], axis=1), train_data[label]
        X_test, y_test = test_data.drop(labels=[label], axis=1), test_data[label]
        X_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)

        problem_type = task_ground_truth_metadata["problem_type"]
        if problem_type in ["multiclass", "binary"]:
            label_map = {k: v for k, v in zip(task_ground_truth_metadata['ordered_class_labels'],
                                              task_ground_truth_metadata['ordered_class_labels_transformed'])}
            y_train = y_train.map(label_map)
            y_test = y_test.map(label_map)
        elif problem_type == "regression":
            pass
        else:
            raise NotImplementedError(f"Problem type not supported yet: {problem_type}")

        if reset_index:
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

        return X_train, y_train, X_test, y_test, preprocessor.feature_metadata

    @property
    def folds(self) -> List[int]:
        return self._zeroshot_context.folds

    def n_folds(self) -> int:
        return len(self.folds)

    def n_datasets(self) -> int:
        return len(self.tids())

    def n_models(self) -> int:
        return len(self.list_models())

    @staticmethod
    def task_name(tid: int, fold: int) -> str:
        return f"{tid}_{fold}"

    @staticmethod
    def task_name_to_tid_and_fold(task_name: str) -> Tuple[int, int]:
        tid, fold = task_name.split("_")
        return tid, fold

    def task_name_from_dataset(self, dataset_name: str, fold: int) -> str:
        return self.task_name(tid=self.dataset_to_tid(dataset_name), fold=fold)

    def evaluate_ensemble(
            self,
            tids: List[int],
            config_names: List[str],
            ensemble_size: int,
            rank: bool = True,
            folds: Optional[List[int]] = None,
            backend: str = "ray",
    ) -> Tuple[np.array, Dict[str, np.array]]:
        """
        :param tids: list of dataset tids to compute errors on.
        :param config_names: list of config to consider for ensembling.
        :param ensemble_size: number of members to select with Caruana.
        :param rank: whether to return ranks or raw scores (e.g. RMSE). Ranks are computed over all base models and
        automl framework.
        :param folds: list of folds that need to be evaluated, use all folds if not provided.
        :return: Tuple:
            2D array of scores whose rows are datasets and columns are folds.
            Dictionary of task_name -> model weights in the ensemble. Model weights are stored in a numpy array,
                with weights corresponding to the order of `config_names`.
        """
        if folds is None:
            folds = self.folds
        tasks = [
            self.task_name(tid=tid, fold=fold)
            for tid in tids
            for fold in folds
        ]
        scorer = self._construct_ensemble_selection_config_scorer(
            datasets=tasks,
            ensemble_size=ensemble_size,
            backend=backend,
        )

        dict_errors, dict_ensemble_weights = scorer.compute_errors(configs=config_names)
        if rank:
            dict_scores = scorer.compute_ranks(errors=dict_errors)
            out = dict_scores
        else:
            out = dict_errors

        out_numpy = np.array([[
            out[self.task_name(tid=tid, fold=fold)
            ] for fold in folds
        ] for tid in tids])

        return out_numpy, dict_ensemble_weights

    def _construct_config_scorer(self,
                                 config_scorer_type: str = 'ensemble',
                                 **config_scorer_kwargs) -> ConfigurationListScorer:
        if config_scorer_type == 'ensemble':
            return self._construct_ensemble_selection_config_scorer(**config_scorer_kwargs)
        elif config_scorer_type == 'single':
            return self._construct_single_best_config_scorer(**config_scorer_kwargs)
        else:
            raise ValueError(f'Invalid config_scorer_type: {config_scorer_type}')

    def _construct_ensemble_selection_config_scorer(self,
                                                    ensemble_size: int = 10,
                                                    backend='ray',
                                                    **kwargs) -> EnsembleSelectionConfigScorer:
        config_scorer = EnsembleSelectionConfigScorer.from_zsc(
            zeroshot_simulator_context=self._zeroshot_context,
            zeroshot_gt=self._ground_truth,
            zeroshot_pred_proba=self._tabular_predictions,
            ensemble_size=ensemble_size,  # 100 is better, but 10 allows to simulate 10x faster
            backend=backend,
            **kwargs,
        )
        return config_scorer

    def _construct_single_best_config_scorer(self, **kwargs) -> SingleBestConfigScorer:
        config_scorer = SingleBestConfigScorer.from_zsc(
            zeroshot_simulator_context=self._zeroshot_context,
            **kwargs,
        )
        return config_scorer


def load(version: str = None, lazy_format=True) -> EvaluationRepository:
    from autogluon_zeroshot.contexts import get_context
    zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = get_context(version).load(load_predictions=True,
                                                                                    lazy_format=lazy_format)
    r = EvaluationRepository(
        zeroshot_context=zsc,
        tabular_predictions=zeroshot_pred_proba,
        ground_truth=zeroshot_gt,
    )
    r = r.force_to_dense(verbose=True)
    return r


# TODO: git shelve ADD BACK
if __name__ == '__main__':
    from autogluon_zeroshot.contexts.context_artificial import load_context_artificial

    with catchtime("loading repo and evaluating one ensemble config"):
        dataset_name = "abalone"
        config_name = "NeuralNetFastAI_r1"
        # repo = EvaluationRepository.load(version="2022_10_13")

        zsc, configs_full, zeroshot_pred_proba, zeroshot_gt = load_context_artificial()
        repo = EvaluationRepository(
            zeroshot_context=zsc,
            tabular_predictions=zeroshot_pred_proba,
            ground_truth=zeroshot_gt,
        )
        tid = repo.dataset_to_tid(dataset_name=dataset_name)
        print(repo.dataset_names()[:3])  # ['abalone', 'ada', 'adult']
        print(repo.tids()[:3])  # [2073, 3945, 7593]

        print(tid)  # 360945
        print(list(repo.list_models_available(tid))[:3])  # ['LightGBM_r181', 'CatBoost_r81', 'ExtraTrees_r33']
        print(repo.eval_metrics(tid=tid, config_names=[config_name],
                                fold=2))  # {'time_train_s': 0.4008138179779053, 'metric_error': 25825.49788, ...
        print(repo.val_predictions(tid=tid, config_name=config_name, fold=2).shape)
        print(repo.test_predictions(tid=tid, config_name=config_name, fold=2).shape)
        print(repo.dataset_metadata(tid=tid))  # {'tid': 360945, 'ttid': 'TaskType.SUPERVISED_REGRESSION
        print(repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name], ensemble_size=5,
                                     backend="native"))  # [[7.20435338 7.04106921 7.11815431 7.08556309 7.18165966 7.1394064  7.03340405 7.11273415 7.07614767 7.21791022]]
        print(repo.evaluate_ensemble(tids=[tid], config_names=[config_name, config_name],
                                     ensemble_size=5, folds=[2], backend="native"))  # [[7.11815431]]
