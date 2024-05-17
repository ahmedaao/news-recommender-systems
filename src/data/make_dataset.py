from typing import List
import os
import io
import pandas as pd


def merge_csv_files_by_rows(directory: str, output_file: str) -> pd.DataFrame:
    """
    Merge multiple CSV files from a directory into a single CSV file.

    Args:
    directory (str): The directory containing CSV files to merge.
    output_file (str): The name of the output CSV file.

    Returns:
    merged_df (pd.DataFrame): The merged pandas dataframe by rows
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

    return merged_df


def merge_csv_files_by_columns(file1_path: str, file2_path: str) -> pd.DataFrame:
    """
    Merge two CSV files based on the 'article_id' column from the first file
    and the 'click_article_id' column from the second file.

    Args:
        file1_path (str): The path to the first CSV file.
        file2_path (str): The path to the second CSV file.

    Returns:
        merged_df (pd.DataFrame): The merged pandas Dataframe by columns
    """
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Rename the column in df2 to match the column name in df1
    df2 = df2.rename(columns={"click_article_id": "article_id"})

    # Merge the two DataFrames based on the 'article_id' column
    merged_df = pd.merge(df1, df2, how="left", on="article_id")

    # Convert the merged DataFrame to a CSV string
    csv_buffer = io.StringIO()
    merged_df.to_csv(csv_buffer, index=False)
    merged_csv = csv_buffer.getvalue()

    return merged_df


def test_function(a, b):
    return a + b
