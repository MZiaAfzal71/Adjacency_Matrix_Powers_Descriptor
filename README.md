# Molecular Descriptor-Based Boiling Point Prediction using Machine Learning

This repository provides Python scripts for predicting the boiling points of small organic molecules using various molecular descriptors and regression models. It introduces a custom descriptor based on the powers of the adjacency matrix and ordered atomic number sequences, and compares it with well-known descriptors.

---

## 📁 Directory Structure

### `Descriptor Generator/`
Contains scripts to generate different types of molecular descriptors from SMILES strings.

- `OurDescriptorGenerator.py`: Generates the novel descriptor proposed in this project.
- `MorganMACCSFingerprintsGenerator.py`: Generates Morgan and MACCS descriptors using RDKit.
- `MordredDescriptorGenerator.py`: Generates Mordred descriptors using the Mordred library.
- `CleanMordred.py`: Cleans Mordred output by replacing NaNs with 0 and removing all-zero columns.
- `CoulombDescriptorGenerator.py`: Computes and flattens the upper triangular Coulomb matrix for descriptor generation.

> ⚠️ **Note:** Please install the required packages (e.g., `rdkit`, `mordred`, `pandas`, etc.) before running these scripts.

---

### `Excel Files/`
Contains raw and generated data files.

- `BoilingPointData5k.xlsx`: Main dataset containing SMILES strings and boiling points for 5432 molecules.
- `OurDescriptor.xlsx`: Custom descriptor generated from adjacency matrix powers and atomic number sequences.
- `MACCSDescriptor.xlsx`, `MorganDescriptor.xlsx`: Fingerprint descriptors generated using RDKit.
- `CoulombMatrixDescriptor.xlsx`: Flattened upper-triangular Coulomb matrix representations.
- `MordredDescriptor.xlsx`: **Not included** due to size—can be generated using `MordredDescriptorGenerator.py`.

> ⚠️ Ensure these files are in the correct path or update the file paths in your scripts accordingly.

---

### `Models/`
Scripts to train and evaluate machine learning models using the descriptors.

- `SupportVectorMachine.py`: Trains a Support Vector Regressor (`kernel='rbf', gamma=0.001`) and evaluates predictions.
- `RandomForest.py`: Trains a Random Forest model with `random_state=42`.
- `XGBoostModel.py`: Uses XGBoost Regressor with fixed seed.

Each script performs:
- 80/20 train-test split (`random_state=42`)
- Training and prediction
- Evaluation using MAE, RMSE, and R²
- Output to an Excel file containing observed and predicted values, and a `"Category"` column indicating `"Train"` or `"Test"` data.

---

## 🔧 Requirements

This project uses Python 3 and the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `rdkit`
- `mordred` (for Mordred descriptors)

Install dependencies using:
```bash
pip install numpy pandas scikit-learn xgboost
# RDKit and Mordred may require conda or extra setup
