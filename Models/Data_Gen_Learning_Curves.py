import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

def plot_learning_curves_row(models_dict, X_dict, y, title, fig_path, scoring='neg_root_mean_squared_error'):
    """
    Plots learning curves for multiple models/descriptors in a single row of subplots.
    Saves both the figure and the learning curve data.

    Parameters:
    - models_dict: dict of {descriptor_name: model}
    - X_dict: dict of {descriptor_name: feature_matrix}
    - y: target array
    - title: overall figure title
    - fig_path: file path to save the plot
    - scoring: cross-validation scoring metric
    """
    n_descriptors = len(models_dict)
    fig, axes = plt.subplots(1, n_descriptors, figsize=(5 * n_descriptors, 4), sharey=True)

    if n_descriptors == 1:
        axes = [axes]  # Ensure axes is iterable

    all_t_sizes, all_t_scores, all_v_scores = [], [], []

    for i, (desc_name, model) in enumerate(models_dict.items()):
        print(f"Processing: {desc_name}")
        X = X_dict[desc_name]

        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=42
        )

        all_t_sizes.append(train_sizes)
        all_t_scores.append(train_scores)
        all_v_scores.append(val_scores)

        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)

        ax = axes[i]
        ax.plot(train_sizes, train_scores_mean, 'o-', label="Train", color='blue')
        ax.plot(train_sizes, val_scores_mean, 'o-', label="Validation", color='orange')
        ax.set_title(desc_name)
        ax.set_xlabel("Train Size")
        if i == 0:
            ax.set_ylabel("RMSE")
        ax.grid(True)
        ax.legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close()
    return all_t_sizes, all_t_scores, all_v_scores


# Load descriptor data
mord_file = pd.read_excel('Excel Files/MordredDescriptor.xlsx')
morg_file = pd.read_excel('Excel Files/MorganDescriptor.xlsx')
macc_file = pd.read_excel('Excel Files/MACCSDescriptor.xlsx')
our_file = pd.read_excel('Excel Files/OurDescriptorWeighted.xlsx')

mord_file.fillna(0, inplace=True)
our_file.fillna(0, inplace=True)

# Extract features and target
X_dict = {
    'MACCS': macc_file.iloc[:, 3:].values,
    'Morgan': morg_file.iloc[:, 3:].values,
    'Mordred': mord_file.iloc[:, 3:].values,
    'OurDescriptor': our_file.iloc[:, 3:].values
}
y = mord_file['Boiling Point'].values  # Shared across descriptors

# Output directory
os.makedirs("Output", exist_ok=True)

# Models and labels
models_to_run = {
    'SVR': SVR(kernel='rbf', gamma=0.001),
    'RF': RandomForestRegressor(random_state=42),
    'XGB': XGBRegressor(random_state=42)
}

# Run for each model
for model_name, base_model in models_to_run.items():
    models_dict = {desc: base_model for desc in X_dict}
    fig_file = f"Output/learning_curves_{model_name.lower()}.png"
    npz_file = f"Output/LC_{model_name.upper()}_Data.npz"

    print(f"\n📊 Generating learning curves for {model_name}...")
    t_sizes, t_scores, v_scores = plot_learning_curves_row(
        models_dict=models_dict,
        X_dict=X_dict,
        y=y,
        title=f"Comparative Learning Curves of {model_name} for Four Descriptor Types",
        fig_path=fig_file
    )
    np.savez(npz_file, t_sizes=t_sizes, t_scores=t_scores, v_scores=v_scores)
    print(f"✅ Saved: {fig_file}, {npz_file}")
