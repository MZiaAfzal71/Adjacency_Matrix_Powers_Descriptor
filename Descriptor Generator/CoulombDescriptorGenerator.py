import pandas as pd
import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import ase
from dscribe.descriptors import CoulombMatrix

def smiles_str_to_rdkit_mol(smiles_str: str) -> rdkit.Chem.Mol:
    """Convert a SMILES string to an RDKit mol object.

    Args:
    - smiles_str (str): A SMILES string representing a molecule.

    Returns:
    - mol (rdkit.Chem.Mol): An RDKit mol object representing the molecule.
    """

    # Convert SMILES string to RDKit mol object
    mol = rdkit.Chem.MolFromSmiles(smiles_str)

    # Add hydrogens to the molecule
    mol = rdkit.Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D structure
    AllChem.UFFOptimizeMolecule(mol)  # Optimize 3D geometry

    return mol


def ase_atoms_to_coulomb_matrix(
        ase_atoms: ase.Atoms,
        log: bool = False
) -> np.ndarray:
    """Convert an ASE Atoms object to a Coulomb matrix and calculate its eigenvalues.

    Args:
        ase_atoms: ASE Atoms object.
        log: Whether to log transform the Coulomb matrix prior to calculating the eigenvalues.

    Returns:
        Eigenvalues of the Coulomb matrix.
    """
    # Create a Coulomb matrix
    coulomb_matrix = CoulombMatrix(
        n_atoms_max=ase_atoms.get_global_number_of_atoms(),
    )

    # Calculate the Coulomb matrix
    coulomb_matrix = coulomb_matrix.create(ase_atoms)
    coulomb_matrix = coulomb_matrix.reshape(
        ase_atoms.get_global_number_of_atoms(),
        ase_atoms.get_global_number_of_atoms())

    # if log:
    #     # Log transform the Coulomb matrix
    #     coulomb_matrix = np.log(coulomb_matrix)
    #
    # # Calculate the eigenvalues of the Coulomb matrix
    # eigenvalues = np.linalg.eigvals(coulomb_matrix)
    return coulomb_matrix  #  coulomb_matrix



input_file = 'Excel Files/BoilingPointData5k.xlsx'
output_file = 'Excel Files/CoulombMatrixDescriptor.csv'

chem_file = pd.read_excel(input_file)

n = len(chem_file)

descriptors = []
for i, sm in chem_file['SMILES'].items():
    try:
        m = smiles_str_to_rdkit_mol(sm)
        ase_atoms = ase.Atoms(
            numbers=[
                atom.GetAtomicNum() for atom in m.GetAtoms()
            ],
            positions=m.GetConformer().GetPositions()
        )
        Col_Mat = ase_atoms_to_coulomb_matrix(ase_atoms)
        sz = len(Col_Mat)
        # vec_len = sz * (sz + 1) // 2
        # vec_len = sz * sz
        Col_Mat = Col_Mat[np.triu_indices(sz)]
        # descriptors.append(Col_Mat.flatten())
        descriptors.append(list(Col_Mat))
    except:
        print(f'{i} did not process!')
    print(f'{i} no of moleculues processed!')

temp_df1 = pd.DataFrame(descriptors)
temp_df1.columns = [f"desc_{i}" for i in range(temp_df1.shape[1])]

base_cols = ['Name', 'SMILES', 'Boiling Point']
temp_df2 = chem_file[base_cols].copy()

# Concatenate the base columns with the descriptor columns
final_df = pd.concat([temp_df2, temp_df1], axis=1)

final_df.to_csv(output_file, index=False)
