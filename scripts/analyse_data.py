import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn.preprocessing import QuantileTransformer
from syne_tune.experiments import load_experiments_df

expname = "YD2H7"
name_filter = lambda path: expname in str(path)
df_tuning = load_experiments_df(path_filter=name_filter)
df_tuning["fold"] = df_tuning.apply(lambda row: 1 + int(row['tuner_name'].split("fold-")[1].split("-")[0]), axis=1)
df_random = df_tuning[df_tuning.scheduler_name == 'RandomSearch']

print(f"Analysing {len(df_random)} configurations")

errors = df_random[['train_error', 'test_error']]
corr = errors.corr()
print(f"Pearson correlation between train and test error: {corr.min().min()}")


errors = df_random[['train_error', 'test_error']]
fig = seaborn.regplot(errors, x='train_error', y='test_error').figure
fig.suptitle("Correlation plots between train and test error")
plt.show()
q_errors = QuantileTransformer().fit_transform(errors)
q_errors = pd.DataFrame(q_errors, columns=errors.columns)
fig = seaborn.regplot(q_errors, x='train_error', y='test_error').figure

fig.suptitle("Correlation plots between train and test ranks")
plt.show()