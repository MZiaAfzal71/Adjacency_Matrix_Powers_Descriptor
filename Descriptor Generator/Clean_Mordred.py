import pandas as pd
import numpy as np


def clean_excel_file(input_file, output_file):
    """Cleans an Excel file by replacing non-numeric values with 0 and removing empty columns."""

    # Load the Excel file
    df = pd.read_excel(input_file)

    # Replace non-numeric values with 0 in columns from index 3 onward (4rd column)
    df.iloc[:, 3:] = df.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Remove columns where all values are 0 (from column index 3 onward)
    df = pd.concat([df.iloc[:, :3], df.iloc[:, 3:].loc[:, (df.iloc[:, 3:] != 0).any(axis=0)]], axis=1)

    # Save the cleaned DataFrame to a new Excel file
    df.to_excel(output_file, index=False)

    print(f"✅ Processed file saved as {output_file}")


# Example Usage
input_file = "Excel Files/MordredDescriptor.xlsx"
output_file = "Excel Files/MordredDescriptor.xlsx"

clean_excel_file(input_file, output_file)
