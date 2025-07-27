import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import os

# Load data
df = pd.read_excel("Output/New_Results.xlsx")

sp_random = 'Split Indices/random_split_42.npz' # path containing random split indices
sp_scaffold = 'Split Indices/scaffold_split_Murcko.npz' # path containing scaffold split indices

split_data = np.load(sp_random)
train_random, val_random, test_random = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']

split_data = np.load(sp_scaffold)
train_scaffold, val_scaffold, test_scaffold = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']

# combine val and test indices

tot_test_random = np.concatenate([val_random, test_random])
tot_test_scaffold = np.concatenate([val_scaffold, test_scaffold])

# Setup
models = ['SVM', 'RF', 'XGB']
descriptors = ['MACCS', 'Morgan', 'Mordred', 'Coulomb', 'WL', 'OurDescriptor']
split_type = ['random', 'scaffold']
splits = ['Train', 'Test']

target = 'Observed'

# Ensure output dirs
os.makedirs("figures/plots", exist_ok=True)
os.makedirs("figures/stats", exist_ok=True)

# Collect performance stats
stats = []

for model in models:
    for desc in descriptors:
        for sp_ty in split_type:
            if sp_ty == 'random':
                subset = df.loc[tot_test_random, :]
                # subset = df.loc[train_random, :]
            else:
                subset = df.loc[tot_test_scaffold, :]
                # subset = df.loc[train_scaffold, :]
            pred_col = f"{model}_{desc}_{sp_ty}_Prediction"
            if pred_col not in subset:
                continue

            y_true = subset[target]
            y_pred = subset[pred_col]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            stats.append({
                "Model": model,
                "Descriptor": desc,
                "Split": sp_ty,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            # Parity Plot
            # plt.figure(figsize=(5, 5))
            # sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
            # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            # plt.xlabel("Observed Boiling Point")
            # plt.ylabel("Predicted")
            # plt.title(f"{model} - {desc} - {split}")
            # plt.tight_layout()
            # plt.savefig(f"figures/plots/parity_{model}_{desc}_{split}.png")
            # plt.close()

# Convert stats to DataFrame
stats_df = pd.DataFrame(stats)
# stats_df.to_csv("figures/stats/performance_summary_new.csv", index=False)
#
# # Boxplot of errors
# df_errors = []
# for model in models:
#     for desc in descriptors:
#         for sp_ty in split_type:
#             if sp_ty == 'random':
#                 subset = df.loc[tot_test_random, :]
#             else:
#                 subset = df.loc[tot_test_scaffold, :]
#             pred_col = f"{model}_{desc}_{sp_ty}_Prediction"
#             if pred_col not in df:
#                 continue
#             error_test = subset[pred_col] - subset[target]
#             df_errors.append(pd.DataFrame({
#                 "Model": model,
#                 "Descriptor": desc,
#                 "Split" : sp_ty,
#                 "Error": error_test
#             }))
#
# error_df = pd.concat(df_errors, ignore_index=True)
# # Create the boxplot with split as a facet
# g = sns.catplot(
#     data=error_df,
#     kind="box",
#     x="Descriptor",
#     y="Error",
#     hue="Model",
#     col="Split",              # Adds separate boxplots for 'random' and 'scaffold'
#     height=5,
#     aspect=1.2,
#     palette="muted",
#     legend_out=False
# )
#
# # Add a shared title above all subplots
# g.fig.suptitle("Prediction Error Distribution by Descriptor and Model on Test Data", fontsize=14)
# g.fig.subplots_adjust(top=0.85)  # Adjust space for suptitle
#
# # Rotate x-axis labels for clarity
# g.set_xticklabels(rotation=45)
# # Save to file
# g.savefig("figures/plots/error_boxplot_by_split.png", bbox_inches="tight")
# plt.close()
#
# Bar plot of MAE for test set
# Set seaborn style
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
# sns.barplot(data=stats_df, x="Descriptor", y="MAE", hue="Model")
g = sns.catplot(
    data=stats_df,
    kind="bar",
    x="Descriptor",
    y="MAE",
    hue="Model",
    col="Split",
    errorbar="sd",
    height=5,
    aspect=1.2,
    palette="muted",
    legend_out=False
)
# Add a shared title above all subplots
g.fig.suptitle("MAE on Test Data", fontsize=14)
g.fig.subplots_adjust(top=0.85)  # Adjust space for suptitle
g.set_titles("Split: {col_name}")
g.set_xticklabels(rotation=45)
# Move legend to top right
# g._legend.set_bbox_to_anchor((0.51, 0.82))
# g._legend.set_loc("upper center")


# g.title("MAE on Test Set")# Optional: draw horizontal line at 0 for reference


plt.tight_layout()
plt.savefig("figures/plots/mae_comparison_test_new.png", bbox_inches="tight")
plt.close()

# for model in models:
#     for desc in descriptors:
#         pred_col = f"{model}_{desc}_Pred"
#
#         y_true = df[target]
#         y_pred = df[pred_col]
#
#         # mae = mean_absolute_error(y_true, y_pred)
#         # rmse = mean_squared_error(y_true, y_pred, squared=False)
#         r2 = r2_score(y_true, y_pred)
#
#         stats.append({
#             "Model": model,
#             "Descriptor": desc,
#             # "Split": split,
#             # "MAE": mae,
#             # "RMSE": rmse,
#             "R²": r2
#         })
#
#
# # Convert stats to DataFrame
# stats_df = pd.DataFrame(stats)
# stats_df.to_csv("r2_summary.csv", index=False)