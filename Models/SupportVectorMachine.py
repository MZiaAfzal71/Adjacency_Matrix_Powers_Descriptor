import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Descriptor files mapping
descriptor_files = {
    'OurDescriptor': 'Excel Files/OurDescriptor.xlsx',
    'Coulomb': 'Excel Files/CoulombMatrixDescriptor.xlsx',
    'Morgan': 'Excel Files/MorganDescriptor.xlsx',
    'MACCS': 'Excel Files/MACCSDescriptor.xlsx',
    'Mordred': 'Excel Files/MordredDescriptor.xlsx'
}

output_dir = 'Output'
os.makedirs(output_dir, exist_ok=True)

# Loop over each descriptor
for desc_name, input_path in descriptor_files.items():
    print(f"\nProcessing descriptor: {desc_name}")

    # Load file (CSV or Excel)
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_excel(input_path)

    data.fillna(0, inplace=True)

    X = data.iloc[:, 3:]  # Features
    y_actual = data['Boiling Point']  # Original target
    max_y = y_actual.max()
    y = y_actual / max_y  # Normalize target for SVR stability

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

    # Initialize and fit SVR
    model = SVR(kernel='rbf', gamma=0.001)
    model.fit(X_train, y_train)

    # Predict and denormalize
    y_pred = model.predict(X) * max_y

    # Evaluation metrics
    print(f"MAE: {mean_absolute_error(y_actual, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_actual, y_pred)):.2f}")
    print(f"R²: {r2_score(y_actual, y_pred):.3f}")

    # Create results DataFrame
    results = pd.DataFrame({
        'Name': data['Name'],
        'SMILES': data['SMILES'],
        'Observed': y_actual,
        'Predicted': y_pred,
        'Category': [''] * len(data)
    })

    results.loc[X_train.index, 'Category'] = 'Train'
    results.loc[X_valid.index, 'Category'] = 'Test'

    # Save output
    output_path = os.path.join(output_dir, f"SVR_{desc_name}.xlsx")
    results.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")


