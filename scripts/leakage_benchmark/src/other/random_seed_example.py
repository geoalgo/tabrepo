"""
Results from Zeroshot Benchmark
                        model  score_test  score_val
0          LightGBM_c1_BAG_L1      0.5109   0.541925  (extra_trees=False)
1          LightGBM_c2_BAG_L1      0.4400   0.541698  (extra_trees=True)
2    NeuralNetTorch_c1_BAG_L1      0.4359   0.547844  (use_batchnorm=False) (unable to reproduce)
3    NeuralNetTorch_c2_BAG_L1      0.5041   0.542743  (use_batchnorm=True) (unable to reproduce)


Results from Default AutoGluon
                      model  score_test  score_val
0        LightGBM_c1_BAG_L1    0.510938   0.541925
1        LightGBM_c2_BAG_L1    0.440000   0.541698
2  NeuralNetTorch_c1_BAG_L1    0.447812   0.518997
3  NeuralNetTorch_c2_BAG_L1    0.483906   0.541950

Result from AutoGluon with safety shuffle
                      model  score_test  score_val (random_seed 43)
0        LightGBM_c1_BAG_L1    0.488125   0.485713
1        LightGBM_c2_BAG_L1    0.408125   0.521467
2  NeuralNetTorch_c1_BAG_L1    0.462656   0.541569
3  NeuralNetTorch_c2_BAG_L1    0.468906   0.546661
                      model  score_test  score_val (random_seed 42)
0        LightGBM_c1_BAG_L1    0.465625   0.520532
1        LightGBM_c2_BAG_L1    0.515781   0.507424
2  NeuralNetTorch_c1_BAG_L1    0.458281   0.520789
3  NeuralNetTorch_c2_BAG_L1    0.489844   0.538438
                      model  score_test  score_val (random_seed 0)
0        LightGBM_c1_BAG_L1    0.510469   0.512347
1        LightGBM_c2_BAG_L1    0.435938   0.531350
2  NeuralNetTorch_c1_BAG_L1    0.437969   0.534211
3  NeuralNetTorch_c2_BAG_L1    0.438437   0.541877
"""
import pandas as pd
import openml
from autogluon.tabular import TabularPredictor


openml_id = 361331


def get_data(tid: int, fold: int):
    # Get Task and dataset from OpenML and return split data
    oml_task = openml.tasks.get_task(tid, download_splits=True, download_data=True,
                                     download_qualities=False, download_features_meta_data=False)

    train_ind, test_ind = oml_task.get_train_test_split_indices(fold)
    X, *_ = oml_task.get_dataset().get_data(dataset_format='dataframe')

    return X.iloc[train_ind, :].reset_index(drop=True), X.iloc[test_ind, :].reset_index(drop=True), oml_task.target_name


def _run():
    l2_train_data, l2_test_data, label = get_data(openml_id, 0)

    # Run AutoGluon
    print("Start running AutoGluon on L2 data.")
    predictor = TabularPredictor(eval_metric='roc_auc', label=label, verbosity=0,
                                 learner_kwargs=dict(random_state=0))
    predictor.fit(
        train_data=l2_train_data,
        hyperparameters={
            "GBM": [
                {"ag_args": {"name_suffix": "_c1"}},
                {"extra_trees": True, "ag_args": {"name_suffix": "_c2"}}
            ],
            "NN_TORCH": [
                {"ag_args": {"name_suffix": "_c1"}},
                {"use_batchnorm": True, "ag_args": {"name_suffix": "_c2"}}
            ],
        },
        fit_weighted_ensemble=False,
        num_stack_levels=0,
        num_bag_folds=8,
        # ag_args_ensemble={"fold_fitting_strategy": "sequential_local"}
    )

    leaderboard = predictor.leaderboard(l2_test_data, silent=True)[['model', 'score_test', 'score_val']].sort_values(by='model').reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(leaderboard)


if __name__ == '__main__':
    _run()
