from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split # , cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# import numpy as np
import pandas as pd

# Load data
# input_file = 'Excel Files/OurDescriptor.xlsx'
# input_file = 'Excel Files/CoulombMatrixDescriptor.xlsx'
# input_file = 'Excel Files/MorganDescriptor.xlsx'
# input_file = 'Excel Files/MACCSDescriptor.xlsx'
input_file = 'Excel Files/MordredDescriptor.xlsx'


# output_file = 'Output/RF_OurDescriptor.xlsx'
# output_file = 'Output/RF_Coulomb.xlsx'
# output_file = 'Output/RF_Morgan.xlsx'
# output_file = 'Output/RF_MACCS.xlsx'
output_file = 'Output/RF_Mordred.xlsx'


chem_file = pd.read_excel(input_file)
chem_file.fillna(0, inplace=True)

X = chem_file.iloc[:, 3:]
y = chem_file['Boiling Point']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

# Train Random Forest model
model = RandomForestRegressor( random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X)

# Print Results
print(f"The mean absolute error is: {mean_absolute_error(y, y_pred)}")
print(f"The R2 score is: {r2_score(y, y_pred)}")
print(f'The mean squared error is {mean_squared_error(y, y_pred)}')

results = pd.DataFrame({'Name' : chem_file['Name'], 'SMILES' : chem_file['SMILES'],
                        'Observed' : chem_file['Boiling Point'], 'Predicted' : y_pred})
results.to_excel(output_file, index=False)

# Cross-validation
# kfold = KFold(n_splits=5)
#
# cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
# print(f"Cross-validated Mean Squared Error: {-np.mean(cv_scores)}")
