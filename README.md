# Molecular Descriptor-Based Boiling Point Prediction using Machine Learning

This repository provides scripts for predicting boiling points of small organic molecules using a variety of molecular descriptors and machine learning models. It includes a custom descriptor based on adjacency matrix powers and atomic number sequences, and compares it against standard molecular representations.

---

## 📁 Directory Structure

### `Descriptor Generator/`
Scripts to compute molecular descriptors from SMILES:

- `OurDescriptorGenerator.py`: Generates the novel descriptor proposed in this project.
- `MorganMACCSFingerprintsGenerator.py`: Generates Morgan and MACCS descriptors using RDKit.
- `MordredDescriptorGenerator.py`: Generates Mordred descriptors using the Mordred library.
- `CleanMordred.py`: Replaces NaN values with 0 and removes columns with all zeros.
- `CoulombDescriptorGenerator.py`: Computes the upper triangular Coulomb matrix as a feature vector.

> 💡 These scripts require installation of cheminformatics libraries like `rdkit` and `mordred`.

---

### `Excel Files/`
Contains the main data files used for training and testing:

- `BoilingPointData5k.xlsx`: Contains SMILES strings and boiling points for 5432 molecules.
- `OurDescriptor.xlsx`, `MACCSDescriptor.xlsx`, `MorganDescriptor.xlsx`, `CoulombMatrixDescriptor.xlsx`: Processed descriptor files.
- `MordredDescriptor.xlsx`: **Not included** due to size. It can be regenerated using the appropriate script.

> ⚠️ Please adjust file paths in the scripts if your file organization differs.

---

### `Models/`
Scripts to train and evaluate regression models:

- `SupportVectorMachine.py`: Uses SVR with `kernel='rbf'` and `gamma=0.001`.
- `RandomForest.py`: Random Forest model with `random_state=42`.
- `XGBoostModel.py`: XGBoost regressor with `random_state=42`.

Each script:
- Performs 80/20 train-test split using `random_state=42`.
- Evaluates model performance using MAE, RMSE, and R².
- Saves predictions to Excel with a `"Category"` column (Train/Test split).

---

## 🔧 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/boiling-point-ml.git
cd boiling-point-ml
pip install -r requirements.txt
