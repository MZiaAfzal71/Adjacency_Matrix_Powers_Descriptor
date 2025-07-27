import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import os

# Load data
df = pd.read_excel("Output/DMPNN Chemprop/Results_DMPNN.xlsx")

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
descriptors = ['', 'pows', 'atomic', 'combined', 'maccs', 'morgan', 'mordred', 'wl']
split_type = ['random', 'scaffold']

target = 'Observed'

# Ensure output dirs
# os.makedirs("figures/plots", exist_ok=True)
os.makedirs("figures/stats", exist_ok=True)

# Collect performance stats
stats = []

for desc in descriptors:
    for sp_ty in split_type:
        if sp_ty == 'random':
            subset = df.loc[tot_test_random, :]
            # subset = df.loc[train_random, :]
        else:
            subset = df.loc[tot_test_scaffold, :]
            # subset = df.loc[train_scaffold, :]
        if desc == '':
            pred_col = f"DMPNN_chemprop_{sp_ty}_Prediction"
        else:
            pred_col = f"DMPNN_chemprop_{desc}_{sp_ty}_Prediction"
        if pred_col not in subset:
            continue

        y_true = subset[target]
        y_pred = subset[pred_col]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        stats.append({
            "Descriptor": desc,
            "Split": sp_ty,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


# Convert stats to DataFrame
stats_df = pd.DataFrame(stats)
stats_df.to_csv("figures/stats/performance_summary_dmpnn.csv", index=False)
