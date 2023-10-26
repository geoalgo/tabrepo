import matplotlib.pyplot as plt
import seaborn
from syne_tune.experiments import load_experiments_df
from tabrepo.loaders import Paths

if __name__ == '__main__':
    expname = "pYAMa"
    csv_filename = Paths.results_root / f"{expname}.csv"

    name_filter = lambda path: expname in str(path)
    df = load_experiments_df(path_filter=name_filter)
    df["fold"] = df.apply(lambda row: 1 + int(row['tuner_name'].split("fold-")[1].split("-")[0]), axis=1)
    print(df.fold.unique())
    for method in ["LocalSearch", "RandomSearch"]:
        for fold in sorted(df.fold.unique()):
            df_sub = df[(df.scheduler_name == method) & (df.fold == fold)]
            fig = seaborn.regplot(df_sub, x='train_error', y='test_error').figure
            plt.plot(
                *df_sub.loc[df_sub.trial_id == 0, ['train_error', 'test_error']].values.flatten(),
                marker="o", color="red"
            )
            fig.suptitle(f"Correlation between train and test error for {method}, fold {fold}")
            plt.savefig(f"Correlation between train and test error for {method}, fold {fold}.png")
            plt.tight_layout()
            plt.show()


    # for fold in sorted(df.fold.unique()):
    #     df_sub = df[(df.fold == fold)]
    #
    #     fig = seaborn.regplot(df_sub, x='train_error', y='test_error').figure
    #     fig.suptitle(f"Correlation between train and test error all methods, fold {fold}")
    #     plt.tight_layout()
    #     plt.savefig(f"Correlation between train and test error all methods, fold {fold}")
    #     plt.show()