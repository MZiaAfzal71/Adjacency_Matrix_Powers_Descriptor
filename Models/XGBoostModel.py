import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Define input and output file mappings for all descriptors
descriptor_files = {
    'OurDescriptor': 'Excel Files/OurDescriptor.xlsx',
    'Coulomb': 'Excel Files/CoulombMatrixDescriptor.csv',
    'Morgan': 'Excel Files/MorganDescriptor.xlsx',
    'MACCS': 'Excel Files/MACCSDescriptor.xlsx',
    'Mordred': 'Excel Files/MordredDescriptor.xlsx'
}

output_dir = 'Output'
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

for desc_name, input_path in descriptor_files.items():
    print(f"\nProcessing descriptor: {desc_name}")

    # Load the data (csv or excel depending on file extension)
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_excel(input_path)

    data.fillna(0, inplace=True)  # Replace missing values with 0

    # Extract features and target
    X = data.iloc[:, 3:]
    y = data['Boiling Point']

    # Split into train/test
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

    # Initialize and fit XGBoost model
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the whole dataset
    y_pred = model.predict(X)

    # Print overall performance
    print(f"MAE: {mean_absolute_error(y, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
    print(f"R²: {r2_score(y, y_pred):.3f}")

    # Create result dataframe with predicted values and categories
    results = pd.DataFrame({
        'Name': data['Name'],
        'SMILES': data['SMILES'],
        'Observed': y,
        'Predicted': y_pred,
        'Category': [''] * len(data)
    })
    results.loc[X_train.index, 'Category'] = 'Train'
    results.loc[X_valid.index, 'Category'] = 'Test'

    # Save to output Excel file
    output_file = os.path.join(output_dir, f"XGB_{desc_name}.xlsx")
    results.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")