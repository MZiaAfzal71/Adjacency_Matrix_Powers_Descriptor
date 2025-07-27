# Boiling Point Prediction Using Adjacency Matrix Powers-Based Molecular Descriptor

This repository contains code and resources for predicting boiling points of small organic molecules using various molecular descriptors and machine learning models. A descriptor based on powers of the adjacency matrix and ordered atomic number sequences is introduced and benchmarked against standard representations.

---

## 📁 Directory Structure

### `Descriptor Generator/`
Scripts for generating molecular descriptors from SMILES:

- `OurWeightedDescriptorGenerator.py`: Generates the proposed weighted descriptor.
- `MorganMACCSFingerprintsGenerator.py`: Computes Morgan and MACCS fingerprints using RDKit.
- `MordredDescriptorGenerator.py`: Generates Mordred descriptors via the `mordred` library.
- `CleanMordred.py`: Cleans Mordred output by replacing NaNs with 0s and removing zero-only columns.
- `CoulombDescriptorGenerator.py`: Constructs descriptor vectors from the upper triangle of the Coulomb matrix.

> 🧪 Dependencies include RDKit, Mordred, NetworkX, and NumPy.

---

### `Excel Files/`
Contains data files used in training and evaluation:

BoilingPointData5k.xlsx
Dataset of 5,432 organic molecules with experimental boiling points collected from:

> Q. Zang, K. Mansouri, A. J. Williams, R. S. Judson, D. G. Allen, W. M. Casey, N. C. Kleinstreuer
> In silico prediction of physicochemical properties of environmental chemicals using molecular fingerprints and machine learning,
> Journal of Chemical Information and Modeling, 57 (2017), pp. 36–49.
> https://doi.org/10.1021/acs.jcim.6b00129

- Descriptor files (feature matrices used in modeling):

- OurDescriptor.xlsx

- MACCSDescriptor.xlsx

- MorganDescriptor.xlsx

- CoulombMatrixDescriptor.xlsx

⚠️ MordredDescriptor.xlsx is not included due to size, but can be regenerated using the scripts in the Models/ directory.

Data splits:

-random_split_42.npz: Predefined random train/validation/test(80:10:10) split

- scaffold_split_Murcko.npz: Scaffold-based train/validation/test(80:10:10) split generated using Murcko scaffolds

📌 Ensure that the splits align with your experiments when running or comparing models.

---

### `Models/`
This folder contains all supporting scripts and code used for training and evaluating machine learning models referenced in the paper. These include:

- Graph kernel-based models

- Classical ML regressors (SVR, Random Forest, XGBoost)

- ChemProp (D-MPNN)

- ChemProp with custom descriptors (via chemprop_interpret)

- Additional experiments run through the Chemeleon framework

⚠️ Note:
These scripts are not directly executable as-is. Paths and configurations must be manually adjusted by the user to match their environment. Currently, Colab or notebook versions are not provided.

---

## 📦 Requirements

Dependencies are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
