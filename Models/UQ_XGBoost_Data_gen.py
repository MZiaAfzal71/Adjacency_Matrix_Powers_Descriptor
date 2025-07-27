import numpy as np
from xgboost import XGBRegressor
from sklearn.utils import resample
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_xgb_bootstrap_ensemble(X, y, n_models=10, **xgb_params):
    models = []
    for seed in range(n_models):
        X_resampled, y_resampled = resample(X, y, random_state=seed)
        model = XGBRegressor(**xgb_params)
        model.fit(X_resampled, y_resampled)
        models.append(model)
    return models


def predict_with_uncertainty_xgb(models, X_test, mean_y, std_y):
    preds = np.stack([model.predict(X_test) * std_y + mean_y for model in models])
    return preds.mean(axis=0), preds.std(axis=0)


# Descriptor files (excluding Coulomb)
descriptor_files = {
    'OurDescriptor': 'Excel Files/OurDescriptorWeighted.xlsx',
    'Morgan': 'Excel Files/MorganDescriptor.xlsx',
    'MACCS': 'Excel Files/MACCSDescriptor.xlsx',
    'Mordred': 'Excel Files/MordredDescriptor.xlsx'
}

# Split configuration
split_files = {
    'random': "Split Indices/random_split_42.npz",
    'scaffold': "Split Indices/scaffold_split_Murcko.npz"
}
# split_name = 'scaffold'
# split_path = split_files[split_name]

split_name = 'random'
split_path = split_files[split_name]

# Output directory
output_dir = 'UQ_Results'
os.makedirs(output_dir, exist_ok=True)

for desc_name, input_path in descriptor_files.items():
    print(f"▶ Processing descriptor: {desc_name}")

    # Load descriptor data
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_excel(input_path)
    data.fillna(0, inplace=True)

    # Extract features and targets
    X_raw = data.iloc[:, 3:].values
    y_raw = data['Boiling Point'].to_numpy().reshape(-1, 1)

    # Conditional scaling
    scale_features = desc_name not in ['MACCS', 'Morgan']
    if scale_features:
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X_raw)
    else:
        X = X_raw

    sc_y = StandardScaler()
    y_scaled = sc_y.fit_transform(y_raw)
    std_y, mean_y = sc_y.scale_[0], sc_y.mean_[0]
    y_scaled = y_scaled.ravel()

    # Load split indices
    split_data = np.load(split_path)
    train_idx, val_idx, test_idx = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']
    combined_test_idx = np.concatenate([val_idx, test_idx])

    # Prepare datasets
    X_train = X[train_idx]
    X_test = X[combined_test_idx]
    y_train = y_scaled[train_idx]
    y_test = y_raw[combined_test_idx].ravel()  # unscaled for evaluation

    # Train ensemble and make predictions
    xgb_models = train_xgb_bootstrap_ensemble(X_train, y_train, n_models=10, random_state=42)
    mean_preds, std_preds = predict_with_uncertainty_xgb(xgb_models, X_test, mean_y, std_y)

    # Save results
    output_path = os.path.join(output_dir, f"{desc_name}_{split_name}_uq_results.npz")
    np.savez(output_path, y_test=y_test, mean_preds=mean_preds, std_preds=std_preds)

    print(f"✅ Saved: {output_path}")
