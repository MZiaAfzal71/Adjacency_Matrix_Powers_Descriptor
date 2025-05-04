import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors

smiles_file = "Excel Files/BoilingPointData5k.xlsx"  # Update with your actual file path
df = pd.read_excel(smiles_file)

# Initialize Mordred Calculator for 2D and 3D descriptors
calc_3d = Calculator(descriptors)  # Compute both 2D & 3D descriptors

# Store Results
descriptors3d_list = []

for index, smiles in enumerate(df['SMILES']):
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))  # Convert SMILES to RDKit Mol
    if mol is None:
        print(f"Invalid SMILES at row {index}: {smiles}")
        continue
    try:
        # Generate 3D Conformer
        mol_3d = Chem.AddHs(mol)  # Add hydrogen atoms
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())  # Generate 3D structure
        AllChem.UFFOptimizeMolecule(mol_3d)  # Optimize 3D geometry

        # Compute 3D Descriptors
        descriptors_3d = calc_3d(mol_3d)

        # Store Results in a dictionary
        descriptors3d_list.append({"SMILES": smiles, **dict(descriptors_3d)})
    except:
        print(f'{index} is not being processed!')
    print(f'{index} number of molecules processsed!')

results_df2 = pd.DataFrame(descriptors3d_list)


base_cols = ['Name', 'SMILES', 'Boiling Point']
temp_df2 = df[base_cols].copy()

# Concatenate the base columns with the descriptor columns
final_df = pd.concat([temp_df2, results_df2], axis=1)

final_df.to_excel("Excel Files/MordredDescriptor.xlsx", index=False)

print(f"Descriptor calculation completed. Results saved to 'MordredDescriptor.xlsx'.")
