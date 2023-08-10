import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.lines import Line2D


def _normalize_performance(ppd, baseline_algorithm, higher_is_better, delta=0.0001):
    def normalize_function(row):
        # https://stats.stackexchange.com/a/178629 scale to -1 = performance of baseline to; 0 = best performance

        if not higher_is_better:
            tmp_row = row.copy() * -1
        else:
            tmp_row = row.copy()

        baseline_performance = tmp_row[baseline_algorithm]
        range_fallback = abs(tmp_row.max() - baseline_performance) <= delta

        if range_fallback:
            mask = abs(tmp_row - baseline_performance) <= delta
            tmp_row[~mask] = -10  # par10 like

            tmp_row[mask] = -1
            return tmp_row

        res = (tmp_row - baseline_performance) / (tmp_row.max() - baseline_performance) - 1

        return res

    return ppd.apply(normalize_function, axis=1)


def _distribution_plot(plot_df, x_col, y_col, x_label, y_label, save_path, baseline_val, baseline_name,
                       dot_name="Performance", figsize=(12, 6), title="", baseline_color="red", significance_map=None,
                       overwrite_xlim=None, xlim_min=-0.1, xlim_max=None, sort_by=None):
    fig, ax = plt.subplots(figsize=figsize)

    xlim = min(list(plot_df.groupby(y_col)[x_col].apply(lambda x: boxplot_stats(x).pop(0)["whislo"])))
    if xlim < 0:
        xlim *= 1.25
    if xlim_min is not None:
        xlim = min(xlim, xlim_min)
    if overwrite_xlim is not None:
        xlim = overwrite_xlim

    left_of_xlim = plot_df.groupby(y_col)[x_col].apply(lambda x: sum(x < xlim)).to_dict()

    if (sort_by is not None) and (sort_by == "median"):
        order = list(plot_df.groupby(y_col)[x_col].apply(lambda x: boxplot_stats(x).pop(0)["med"]).sort_values(
            ascending=False).index)
    else:
        order = None

    sns.boxplot(data=plot_df, y=y_col, x=x_col, showfliers=False, order=order)
    sns.stripplot(data=plot_df, y=y_col, x=x_col, color="black", order=order)

    ax.axvline(x=baseline_val, c=baseline_color)
    plt.xlabel(x_label)
    yticks = [item.get_text() for item in ax.get_yticklabels()]

    new_yticks = []
    ot_used = False
    for ytick in yticks:
        s_yt = ytick
        if (significance_map is not None):
            s_yt = s_yt + "$^{" + significance_map[ytick] + "}$"

        if left_of_xlim[ytick]:
            ot_used = True
            s_yt = s_yt + f" [{left_of_xlim[ytick]}]"

        new_yticks.append(s_yt)

    ax.set_yticklabels(new_yticks)

    plt.xlim(left=xlim)
    if xlim_max is not None:
        plt.xlim(right=xlim_max)
    plt.legend(handles=[
        Line2D([0], [0], label=baseline_name, color=baseline_color),
        Line2D([], [], marker="o", color="black", label=f"{dot_name} for one Dataset", linestyle="None")
    ],
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
        fancybox=True, shadow=True, ncol=2
    )
    plt.ylabel(y_label + " [#Points Left of x-axis limit]" if ot_used else y_label)
    plt.tight_layout()

    plt.savefig(save_path)

    # plt.xlabel(x_label + " | " + title)
    # plt.show()
    plt.close()


def normalized_improvement_distribution_plot(performance_per_dataset, maximize_metric, baseline_name, output_path):
    plot_df = _normalize_performance(performance_per_dataset, baseline_name, maximize_metric)
    plot_df.drop(columns=[baseline_name], inplace=True)
    plot_df = plot_df.reset_index().melt(id_vars=["Dataset"], value_name="Normalized Improvement", var_name="Algorithm")
    _distribution_plot(plot_df,
                       x_col="Normalized Improvement", y_col="Algorithm",
                       x_label="Normalized Improvement", y_label="Algorithms",
                       save_path=output_path,
                       baseline_val=-1,
                       overwrite_xlim=-1.5,
                       xlim_max=0.1,
                       baseline_name="KNN",
                       dot_name="Normalized Improvement for one Metric",
                       sort_by="median",
                       figsize=(12, 10))