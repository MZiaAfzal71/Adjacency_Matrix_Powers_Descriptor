import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os

# Plotting functions
def plot_uncertainty_vs_error(ax, std_preds, errors, label, color):
    ax.scatter(std_preds, errors, alpha=0.5, label=label, color=color)
    ax.set_xlabel("Predicted Uncertainty (Std)")
    ax.set_ylabel("Absolute Error")
    ax.grid(True)

def plot_calibration_curve(ax, y_true, y_pred_mean, y_pred_std, n_bins=10, label=None, color=None):
    bins = np.linspace(y_pred_std.min(), y_pred_std.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred_std, bins) - 1

    bin_centers = []
    bin_mae = []

    for b in range(n_bins):
        idxs = bin_indices == b
        if np.sum(idxs) > 0:
            bin_centers.append(np.mean(y_pred_std[idxs]))
            bin_mae.append(mean_absolute_error(y_true[idxs], y_pred_mean[idxs]))

    ax.plot(bin_centers, bin_mae, 'o-', label=label, color=color)
    ax.plot(bin_centers, bin_centers, 'k--', linewidth=1)
    ax.set_xlabel("Predicted Std (Uncertainty)")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.grid(True)

# Coverage computation
def compute_coverage(y_true, y_pred_mean, y_pred_std, alpha=0.05):
    z = 1.96  # for 95% CI
    lower = y_pred_mean - z * y_pred_std
    upper = y_pred_mean + z * y_pred_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return coverage

# Config
descriptor_list = ['OurDescriptor', 'Morgan', 'MACCS', 'Mordred']
splits = ['random', 'scaffold']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
results_dir = 'UQ_Results'
coverage_file = os.path.join(results_dir, 'coverage_results.txt')

# Prepare plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Open file to save coverage
with open(coverage_file, 'w') as f:
    for split_idx, split in enumerate(splits):
        for desc_idx, desc in enumerate(descriptor_list):
            # Load data
            file_path = os.path.join(results_dir, f"{desc}_{split}_uq_results.npz")
            data = np.load(file_path)
            y_test = data['y_test']
            mean_preds = data['mean_preds']
            std_preds = data['std_preds']

            # Compute error and coverage
            errors = np.abs(mean_preds - y_test)
            coverage = compute_coverage(y_test, mean_preds, std_preds)

            # Save coverage to file
            f.write(f"{desc} ({split}): Coverage = {coverage*100:.2f}%\n")

            # Plot uncertainty vs error
            plot_uncertainty_vs_error(
                axes[split_idx],
                std_preds, errors,
                label=desc,
                color=colors[desc_idx]
            )

            # Plot calibration curve
            plot_calibration_curve(
                axes[split_idx + 2],
                y_test, mean_preds, std_preds,
                label=desc,
                color=colors[desc_idx]
            )

# Legends and Titles
axes[0].set_title("Uncertainty vs Error (Random Split)")
axes[1].set_title("Uncertainty vs Error (Scaffold Split)")
axes[2].set_title("Calibration Curve (Random Split)")
axes[3].set_title("Calibration Curve (Scaffold Split)")

axes[0].legend()
axes[1].legend()
axes[2].legend()
axes[3].legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "uncertainty_analysis_summary.png"))
plt.show()
