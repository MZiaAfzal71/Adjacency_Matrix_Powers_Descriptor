# Boiling Point Prediction Using Adjacency Matrix Powers-Based Molecular Descriptor

This repository contains code and resources for predicting boiling points of small organic molecules using various molecular descriptors and machine learning models. A descriptor based on powers of the adjacency matrix and ordered atomic number sequences is introduced and benchmarked against standard representations.

---

## 📁 Directory Structure

### `Descriptor Generator/`
Scripts for generating molecular descriptors from SMILES:

- `OurDescriptorGenerator.py`: Generates the proposed descriptor.
- `MorganMACCSFingerprintsGenerator.py`: Computes Morgan and MACCS fingerprints using RDKit.
- `MordredDescriptorGenerator.py`: Generates Mordred descriptors via the `mordred` library.
- `CleanMordred.py`: Cleans Mordred output by replacing NaNs with 0s and removing zero-only columns.
- `CoulombDescriptorGenerator.py`: Constructs descriptor vectors from the upper triangle of the Coulomb matrix.

> 🧪 Dependencies include RDKit, Mordred, NetworkX, and NumPy.

---

### `Excel Files/`
Contains data files used in training and evaluation:

- `BoilingPointData5k.xlsx`: Boiling points for 5432 molecules collected from:
  
  > **Q. Zang, K. Mansouri, A. J. Williams, R. S. Judson, D. G. Allen, W. M. Casey, N. C. Kleinstreuer,**  
  > *In silico prediction of physicochemical properties of environmental chemicals using molecular fingerprints and machine learning*,  
  > Journal of Chemical Information and Modeling, 57 (2017), pp. 36–49. [https://doi.org/10.1021/acs.jcim.6b00129](https://doi.org/10.1021/acs.jcim.6b00129)

- `OurDescriptor.xlsx`, `MACCSDescriptor.xlsx`, `MorganDescriptor.xlsx`, `CoulombMatrixDescriptor.xlsx`: Generated feature matrices for each representation.
- ⚠️ `MordredDescriptor.xlsx` is not included due to size, but can be regenerated using the provided scripts.

---

### `Models/`
Machine learning models used to predict boiling points:

- `SupportVectorMachine.py`: Implements SVR (`kernel='rbf'`, `gamma=0.001`).
- `RandomForest.py`: Uses a Random Forest Regressor with `random_state=42`.
- `XGBoostModel.py`: Implements XGBoost with `random_state=42`.

Each model:
- Performs an 80/20 train-test split using `sklearn` with `random_state=42`.
- Outputs MAE, RMSE, R² scores, and saves predictions to Excel including a `Category` column (Train/Test).

---

## 📦 Requirements

Dependencies are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
