# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Descriptor names (assumed in order)
# # descriptors = ["OurDescriptor", "Coulomb", "Morgan", "MACCS", "Mordred", "WL Kernel"]
# descriptors = ["MACCS", "Morgan", "Mordred", "OurDescriptor"]
#
# # Load data
# data_SVR = np.load('LC_SVR_Data.npz')
# data_RF = np.load('LC_RF_Data.npz')
# data_XGB = np.load('LC_XGB_Data.npz')
#
# training_sizes = data_SVR['t_sizes'][0]
#
# # Extract and average over CV folds
# models_data = {
#     'SVR': (-np.mean(data_SVR['t_scores'], axis=2), -np.mean(data_SVR['v_scores'], axis=2)),
#     'RF': (-np.mean(data_RF['t_scores'], axis=2), -np.mean(data_RF['v_scores'], axis=2)),
#     'XGB': (-np.mean(data_XGB['t_scores'], axis=2), -np.mean(data_XGB['v_scores'], axis=2)),
# }
#
# # Colors and styles
# colors = {'SVR': 'tab:blue', 'RF': 'tab:green', 'XGB': 'tab:orange'}
# linestyles = {'Train': '-', 'Validation': '--'}
#
# # Plotting
# sns.set(style="whitegrid")
# fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
#
# for i, desc in enumerate(descriptors):
#     # l = i % 2
#     # m = i // 2
#     ax = axes[i]
#     for model_name, (train_scores, val_scores) in models_data.items():
#         ax.plot(training_sizes, train_scores[i], 'o', label=f'{model_name} Train', color=colors[model_name], linestyle=linestyles['Train'])
#         ax.plot(training_sizes, val_scores[i], '^', label=f'{model_name} Val', color=colors[model_name], linestyle=linestyles['Validation'])
#
#     ax.set_title(desc, fontsize=10)
#     ax.set_xlabel("Training Size")
#     if i == 0:
#         ax.set_ylabel("RMSE")
#     ax.tick_params(axis='x', labelrotation=45)
#     ax.grid(True)
#
# # Shared legend
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper left', ncol=3, bbox_to_anchor=(0, 1), frameon=False)
# fig.suptitle("Learning Curves for All Descriptors across Models", fontsize=14)
# fig.tight_layout()
# # bbox_to_anchor = (0.2, 1)rect=[0, 0, 1, 1]
# # Save and show
# plt.savefig("learning_curves_all_descriptors.png", dpi=500)
# plt.show()
#
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Descriptor names
descriptors = ["MACCS", "Morgan", "Mordred", "OurDescriptor"]

# Load data
data_SVR = np.load('LC_SVR_Data.npz')
data_RF = np.load('LC_RF_Data.npz')
data_XGB = np.load('LC_XGB_Data.npz')

training_sizes = data_SVR['t_sizes'][0]

# Extract and average over CV folds
models_data = {
    'SVR': (-np.mean(data_SVR['t_scores'], axis=2), -np.mean(data_SVR['v_scores'], axis=2)),
    'RF': (-np.mean(data_RF['t_scores'], axis=2), -np.mean(data_RF['v_scores'], axis=2)),
    'XGB': (-np.mean(data_XGB['t_scores'], axis=2), -np.mean(data_XGB['v_scores'], axis=2)),
}

# Colors and styles
colors = {'SVR': 'tab:blue', 'RF': 'tab:green', 'XGB': 'tab:orange'}
linestyles = {'Train': '-', 'Validation': '--'}

# Plotting
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for i, desc in enumerate(descriptors):
    ax = axes[i]
    for model_name, (train_scores, val_scores) in models_data.items():
        ax.plot(training_sizes, train_scores[i], 'o', label=f'{model_name} Train', color=colors[model_name], linestyle=linestyles['Train'])
        ax.plot(training_sizes, val_scores[i], 's', label=f'{model_name} Val', color=colors[model_name], linestyle=linestyles['Validation'])

    ax.set_title(desc, fontsize=11)
    ax.grid(True)

# Shared x and y labels
fig.text(0.5, 0.02, 'Training Size', ha='center', fontsize=12)
fig.text(0.02, 0.5, 'RMSE', va='center', rotation='vertical', fontsize=12)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1), frameon=False)
fig.suptitle("Learning Curves for All Descriptors across Models", fontsize=14, y=1.12)

fig.tight_layout(rect=[0.03, 0.03, 1, 1])
plt.savefig("learning_curves_all_descriptors_grid.png", dpi=500)
plt.show()
