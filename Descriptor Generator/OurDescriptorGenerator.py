import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

def smiles_to_adjacency_matrix(smiles):
    """Converts a SMILES string to adjacency matrix and extracts atomic numbers."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None  # Invalid SMILES
    mol = Chem.AddHs(mol)  # Add explicit hydrogens

    # Adjacency Matrix (A)
    A = rdmolops.GetAdjacencyMatrix(mol)

    # Atomic Numbers sorted in descending order
    atomic_numbers = sorted([atom.GetAtomicNum() for atom in mol.GetAtoms()], reverse=True)

    return A, atomic_numbers

def compute_sums_of_powers(adj_matrix, n):
    """Computes the sum of all elements of A, A^2, ..., A^n."""
    sums = []
    log_adj_matrix = np.log1p(adj_matrix)
    sums.append(np.sum(log_adj_matrix))
    matrix_prod = np.log1p(log_adj_matrix.copy())

    for i in range(n-1):
        matrix_prod = np.matmul(matrix_prod, log_adj_matrix)
        # Apply log1p transform to avoid log(0) errors
        matrix_prod = np.log1p(matrix_prod)
        sums.append(np.sum(matrix_prod))  # Sum of all elements

    return sums

def process_excel_file(input_file, output_file, n):
    """Reads input Excel file, computes adjacency matrix sums + atomic numbers, and writes output."""
    df = pd.read_excel(input_file)

    sum_columns = []  # Stores A^n sums
    atomic_number_columns = []  # Stores atomic numbers in descending order
    atom_sizes = []  # Stores number of atoms per molecule

    for index, row in df.iterrows():
        smiles = row['SMILES']
        A, atomic_numbers = smiles_to_adjacency_matrix(smiles)

        if A is None or atomic_numbers is None:
            sum_columns.append([None] * n)
            atomic_number_columns.append([])
            atom_sizes.append(0)
        else:
            sum_columns.append(compute_sums_of_powers(A, n))  # Compute A^1 to A^n
            atomic_number_columns.append(atomic_numbers)
            atom_sizes.append(len(atomic_numbers))  # Store size for alignment

    # Convert sums list to DataFrame
    sum_col_names = [f'Sum_A^{i+1}' for i in range(n)]
    sum_df = pd.DataFrame(sum_columns, columns=sum_col_names)

    # Convert atomic numbers list to DataFrame (padded to max length)
    max_atom_length = max(atom_sizes)
    atom_col_names = [f'Atomic_Num_{i+1}' for i in range(max_atom_length)]

    # Pad shorter lists with NaN to match length
    atomic_number_df = pd.DataFrame([x + [np.nan] * (max_atom_length - len(x)) for x in atomic_number_columns], columns=atom_col_names)

    # Merge both DataFrames with the original
    df = pd.concat([df, sum_df, atomic_number_df], axis=1)

    # Save the final DataFrame
    df.to_excel(output_file, index=False)
    print(f"✅ Processed file saved as {output_file}")

# Example usage
input_file = "Excel Files/BoilingPointData5k.xlsx"
output_file = "Excel Files/OurDescriptor.xlsx"
n = 50  # Number of A^n powers to compute

process_excel_file(input_file, output_file, n)
