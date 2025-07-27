import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
font1 = {'family': 'serif', 'weight': 'bold', 'size': 11}
font2 = {'family': 'serif', 'weight': 'bold', 'size': 13}

# Load data
df = pd.read_excel("Output/New_Results.xlsx")

# Load split indices
sp_random = np.load('Split Indices/random_split_42.npz')
sp_scaffold = np.load('Split Indices/scaffold_split_Murcko.npz')

train_random = sp_random['train_idx']
val_random = sp_random['val_idx']
test_random = sp_random['test_idx']
tot_test_random = np.concatenate([val_random, test_random])

train_scaffold = sp_scaffold['train_idx']
val_scaffold = sp_scaffold['val_idx']
test_scaffold = sp_scaffold['test_idx']
tot_test_scaffold = np.concatenate([val_scaffold, test_scaffold])

# Define settings
models = ['SVM', 'RF', 'XGB']                     # Supported models
enabled_models = ['RF']                           # 🔧 Change to ['SVM', 'RF', 'XGB'] to enable all
splits = ['random', 'scaffold']
descriptors = ['MACCS', 'Morgan', 'Mordred', 'Coulomb', 'WL', 'OurDescriptor']
colors = {'train': 'blue', 'test': 'orange'}

# Observed data range
min_data = df['Observed'].min()
max_data = df['Observed'].max()

# Create subplots
fig, axes = plt.subplots(2, len(descriptors), sharey=True, figsize=(15, 8))
axes = axes.flatten()

# Iterate over descriptors
for idx, desc in enumerate(descriptors):
    for split_idx, split in enumerate(splits):
        row_offset = split_idx * len(descriptors)
        ax = axes[row_offset + idx]

        # Get indices
        if split == 'random':
            train_idx, test_idx = train_random, tot_test_random
        else:
            train_idx, test_idx = train_scaffold, tot_test_scaffold

        train_obs = df.loc[train_idx, 'Observed']
        test_obs = df.loc[test_idx, 'Observed']

        # Try models in priority order (use first available)
        for model in enabled_models:
            col_name = f"{model}_{desc}_{split}_Prediction"
            if col_name in df.columns:
                train_pred = df.loc[train_idx, col_name]
                test_pred = df.loc[test_idx, col_name]
                break
        else:
            continue  # Skip if no model found

        # Plot
        sns.scatterplot(x=train_obs, y=train_pred, alpha=1, ax=ax, color=colors['train'], label="Train")
        sns.scatterplot(x=test_obs, y=test_pred, alpha=0.4, ax=ax, color=colors['test'], label="Test")
        ax.plot([min_data, max_data], [min_data, max_data], 'r--')

        # Aesthetics
        ax.spines[['right', 'top']].set_visible(False)
        if split == 1:
            ax.set_xlabel('')

        if idx == 0:
            ylabel = 'Random' if split == 0 else 'Scaffold'
            ax.set_ylabel(ylabel, fontdict=font1)
        else:
            ax.set_ylabel('')

        if split == 0:
            ax.set_title(desc, fontdict=font2)

# Add shared labels
fig.text(0.5, 0.04, r'Observed Boiling Points $^{\circ}$C', ha='center', fontdict=font2)
fig.text(0.05, 0.5, r'Predicted Boiling Points $^{\circ}$C', va='center', rotation='vertical', fontdict=font2)

plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig("Output/RF_Plots_Looped.png", bbox_inches="tight", dpi=300)
plt.show()
