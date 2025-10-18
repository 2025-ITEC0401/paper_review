# -*- coding: utf-8 -*-
import pandas as pd
import argparse

def merge_csv_files(file1_path, file2_path, output_path):
    """
    Merge two CSV files by adding the 2nd column from file1 to the end of file2
    
    Args:
        file1_path: Path to the first CSV file (to get 2nd column from)
        file2_path: Path to the second CSV file (to add column to)
        output_path: Path to save the result CSV file
    """
    # Load CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    
    # Get the 2nd column from the first file (index 1)
    second_column = df1.iloc[:, 1]
    column_name = df1.columns[1]
    
    print(f"\nColumn to add: '{column_name}'")
    print(f"Column length: {len(second_column)}")
    
    # Handle different row counts
    if len(second_column) != len(df2):
        print(f"\nWarning: The two files have different row counts!")
        print(f"File 1 2nd column rows: {len(second_column)}")
        print(f"File 2 rows: {len(df2)}")
        
        # Adjust to shorter length
        min_length = min(len(second_column), len(df2))
        df2 = df2.iloc[:min_length]
        second_column = second_column.iloc[:min_length]
        print(f"Adjusted both files to {min_length} rows")
    
    # Handle duplicate column names
    if column_name in df2.columns:
        new_column_name = f"{column_name}_from_file1"
        print(f"\nColumn name conflict! '{column_name}' -> '{new_column_name}'")
        column_name = new_column_name
    
    # Add column to the end of file2
    df2[column_name] = second_column.values
    
    # Save result
    df2.to_csv(output_path, index=False)
    print(f"\nResult saved to '{output_path}'")
    print(f"Final shape: {df2.shape}")
    print(f"\nResult file columns: {list(df2.columns)}")
    
    return df2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two CSV files.')
    parser.add_argument('--file1', type=str, required=True, 
                        help='Path to the first CSV file (to get 2nd column from)')
    parser.add_argument('--file2', type=str, required=True,
                        help='Path to the second CSV file (to add column to)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the result CSV file')
    
    args = parser.parse_args()
    
    merge_csv_files(args.file1, args.file2, args.output)
