from typing import List
import os
import pandas as pd


def merge_csv_files(directory: str, output_file: str) -> None:
    """
    Merge multiple CSV files from a directory into a single CSV file.

    Args:
    directory (str): The directory containing CSV files to merge.
    output_file (str): The name of the output CSV file.

    Returns:
    None
    """
    # List to store individual DataFrames
    all_dataframes: List[pd.DataFrame] = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)
            # Add the DataFrame to the list
            all_dataframes.append(df)

    # Merge all DataFrames into one
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)
