import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# -------------------------
# Descriptor and split settings
# -------------------------
descriptor_files = {
    # 'Adjacency_Pow' : 'Excel Files/apow_descriptor.csv',
    # 'Atomic_No' : 'Excel Files/atomic_descriptor.csv' #,
    'OurDescriptor': 'Excel Files/OurDescriptorWeighted.xlsx',
    'Coulomb': 'Excel Files/CoulombMatrixDescriptor.xlsx',
    'Morgan': 'Excel Files/MorganDescriptor.xlsx',
    'MACCS': 'Excel Files/MACCSDescriptor.xlsx',
    'Mordred': 'Excel Files/MordredDescriptor.xlsx'
}

split_files = {
    'random': "Split Indices/random_split_42.npz",
    'scaffold': "Split Indices/scaffold_split_Murcko.npz"
}

output_dir = 'Output'
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Loop over descriptors
# -------------------------
for desc_name, input_path in descriptor_files.items():
    print(f"\n🧬 Processing descriptor: {desc_name}")

    # Load descriptor data
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_excel(input_path)

    data.fillna(0, inplace=True)  # Handle NaNs

    # Extract features and targets
    X_raw = data.iloc[:, 3:].values
    y_raw = data['Boiling Point'].to_numpy().reshape(-1, 1)

    # Apply scaling conditionally
    scale_features = desc_name not in ['MACCS', 'Morgan']

    if scale_features:
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X_raw)
    else:
        X = X_raw  # No scaling

    sc_y = StandardScaler()
    y_scaled = sc_y.fit_transform(y_raw)
    std_y, mean_y = sc_y.scale_[0], sc_y.mean_[0]
    y_scaled = y_scaled.ravel()  # Flatten to 1D

    # -------------------------
    # Loop over both splits
    # -------------------------
    for split_name, split_path in split_files.items():
        print(f"📂 Using split: {split_name}")

        # Load split indices
        split_data = np.load(split_path)
        train_idx, val_idx, test_idx = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']
        combined_test_idx = np.concatenate([val_idx, test_idx])

        # Prepare train/test sets
        X_train = X[train_idx]
        X_test = X[combined_test_idx]
        y_train = y_scaled[train_idx]
        y_test = y_scaled[combined_test_idx]

        # Train XGB model
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)

        # Predict on full set
        y_pred_scaled = model.predict(X)
        y_true = y_scaled * std_y + mean_y
        # y_pred_scaled = model.predict(X_test)
        # y_true = y_test * std_y + mean_y
        y_pred = y_pred_scaled * std_y + mean_y

        # Report performance
        print(f"🔍 MAE:  {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"🔍 RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        print(f"🔍 R²:   {r2_score(y_true, y_pred):.3f}")

        # Construct result dataframe
        results = pd.DataFrame({
            'Name': data['Name'],
            'SMILES': data['SMILES'],
            'Observed': y_true,
            'Predicted': y_pred,
            'Category': [''] * len(data)
        })

        results.loc[train_idx, 'Category'] = 'Train'
        results.loc[combined_test_idx, 'Category'] = 'Test'

        # Save to Excel
        output_file = os.path.join(output_dir, f"SVM_{desc_name}_{split_name}.xlsx")
        results.to_excel(output_file, index=False)
        print(f"✅ Saved to: {output_file}")
