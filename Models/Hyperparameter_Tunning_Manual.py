import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from contextlib import redirect_stdout
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
input_csv = "Excel Files/MorganDescriptor.xlsx"
output_dir = "Output/XGB_Morgan_CV"
os.makedirs(output_dir, exist_ok=True)

n_splits = 5
random_state = 42
cv_metric_file = os.path.join(output_dir, "cv_summary.txt")
npz_save_path = os.path.join(output_dir, "cv_scores.npz")

# -----------------------------
# Load and Preprocess Data
# -----------------------------
data = pd.read_excel(input_csv)
data.fillna(0, inplace=True)

X_raw = data.iloc[:, 3:].values
y_raw = data['Boiling Point'].to_numpy().reshape(-1, 1)

# Normalize features and target
sc_X = StandardScaler()
X = sc_X.fit_transform(X_raw)

sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y_raw).ravel()
std_y, mean_y = sc_y.scale_[0], sc_y.mean_[0]


# -----------------------------
# Cross-Validation Runner
# -----------------------------
def run_cv_xgb(param_grid, X, y_scaled, std_y, mean_y, save_txt, save_npz):
    results = []

    with open(save_txt, 'w') as f:
        with redirect_stdout(f):
            for eta in param_grid['eta']:
                for n_estimators in param_grid['n_estimators']:
                    for max_depth in param_grid['max_depth']:
                        for min_child_weight in param_grid['min_child_weight']:
                            for subsample in param_grid['subsample']:
                                for colsample_bytree in param_grid['colsample_bytree']:

                                    print(f"\n🔧 Hyperparameters: eta={eta}, n_estimators={n_estimators}, "
                                          f"max_depth={max_depth}, min_child_weight={min_child_weight}, "
                                          f"subsample={subsample}, colsample_bytree={colsample_bytree}")

                                    model = XGBRegressor(
                                        eta=eta,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_child_weight=min_child_weight,
                                        gamma=0,
                                        reg_lambda=1,
                                        reg_alpha=1,
                                        subsample=subsample,
                                        colsample_bytree=colsample_bytree,
                                        random_state=random_state,
                                        verbosity=0
                                    )

                                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                                    maes, rmses, r2s = [], [], []

                                    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                                        X_train, X_test = X[train_idx], X[test_idx]
                                        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)

                                        # Rescale predictions
                                        y_pred_rescaled = y_pred * std_y + mean_y
                                        y_test_rescaled = y_test * std_y + mean_y

                                        # Metrics
                                        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                                        rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
                                        r2_val = r2_score(y_test_rescaled, y_pred_rescaled)

                                        print(f"  Fold-{fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2_val:.3f}")
                                        maes.append(mae)
                                        rmses.append(rmse)
                                        r2s.append(r2_val)

                                    mean_mae, std_mae = np.mean(maes), np.std(maes)
                                    mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
                                    mean_r2, std_r2 = np.mean(r2s), np.std(r2s)

                                    print(f"📊 Result: MAE={mean_mae:.2f}±{std_mae:.2f}, "
                                          f"RMSE={mean_rmse:.2f}±{std_rmse:.2f}, "
                                          f"R²={mean_r2:.3f}±{std_r2:.3f}")

                                    results.append({
                                        'params': {
                                            'eta': eta,
                                            'n_estimators': n_estimators,
                                            'max_depth': max_depth,
                                            'min_child_weight': min_child_weight,
                                            'subsample': subsample,
                                            'colsample_bytree': colsample_bytree,
                                        },
                                        'maes': maes,
                                        'rmses': rmses,
                                        'r2s': r2s
                                    })

    # Save metrics
    np.savez_compressed(
        save_npz,
        results=results
    )
    print(f"\n✅ Saved CV results to: {save_npz}")


# -----------------------------
# Define Hyperparameter Grid
# -----------------------------
param_grid = {
    'eta': [0.09, 0.08],
    'n_estimators': [500, 700],
    'max_depth': [6, 10],
    'min_child_weight': [1, 0.8],
    'subsample': [1, 0.8],
    'colsample_bytree': [1, 0.8]
}

# -----------------------------
# Run and Save Results
# -----------------------------
run_cv_xgb(param_grid, X, y_scaled, std_y, mean_y, cv_metric_file, npz_save_path)
