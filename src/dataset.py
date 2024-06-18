# Import packages
from typing import List
import os
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def concat_csv_files(directory: str, output_file: str) -> pd.DataFrame:
    """
    Concat multiple CSV files from a directory into a single CSV file.

    Args:
    directory (str): The directory containing CSV files to concat.
    output_file (str): The name of the output CSV file.

    Returns:
    concated_df (pd.DataFrame): The concated pandas dataframe
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
    concated_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the merged DataFrame to a CSV file
    concated_df.to_csv(output_file, index=False)

    return concated_df


def merge_csv_files(file1_path: str, file2_path: str, output_file: str) -> pd.DataFrame:
    """
    Merge two CSV files based on the 'article_id' column from the first file
    and the 'click_article_id' column from the second file.

    Args:
        file1_path (str): The path to the first CSV file.
        file2_path (str): The path to the second CSV file.
        output_file (str): The name of the output CSV file.

    Returns:
        merged_df (pd.DataFrame): The merged pandas Dataframe by columns
    """
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge the two DataFrames based on the 'article_id' column
    merged_df = pd.merge(
        df1, df2, how="left", left_on="click_article_id", right_on="article_id"
    )

    merged_df = merged_df.drop(columns=["click_article_id"])

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_file, index=False)

    return merged_df


def closest_articles(
    model_dir_path: str, article_id: int, nb_closest_articles: int
) -> dict:
    """
    Find the closest articles to a given article based on cosine similarity.

    Parameters:
    model_dir_path (str): The directory path where the articles embeddings are.
    article_id (int): The ID of the article for which the closest articles are to be found.
    nb_closest_articles (int): The number of closest articles to retrieve.

    Returns:
    dict: A dictionary containing:
        - "indices": The indices of the closest articles.
        - "cosine_similarities": The cosine similarity values of the closest articles.
    """
    # Deserialize pickle object
    with open(model_dir_path + "articles_embeddings.pickle", "rb") as f:
        articles_embeddings = cPickle.load(f)

    row = articles_embeddings[article_id].reshape(1, -1)
    cosine_similarities = cosine_similarity(row, articles_embeddings).flatten()

    sorted_indices = np.argsort(-cosine_similarities)[1:][:nb_closest_articles]
    sorted_cosine_similarities = np.sort(cosine_similarities)[::-1][1:][
        :nb_closest_articles
    ]

    results = {
        "indices": sorted_indices,
        "cosine_similarities": sorted_cosine_similarities,
    }

    return results


def filter_by_user_counts(df: pd.DataFrame, number_of_articles: int):
    """
    DataFrame to keep users who have clicked on a min number of articles.

    Args:
        df (pd.DataFrame): The input DataFrame
        min_click_articles (int): Apply filter to this number of articles

    Returns:
        pd.DataFrame: A filtered DataFrame containing only users who have clicked on at least `min_click_articles` articles.
    """
    user_counts = df["user_id"].value_counts()
    filtered_users = user_counts[user_counts > number_of_articles].index
    filtered_df = df[df["user_id"].isin(filtered_users)]

    return filtered_df


def articles_not_clicked_by_user(df: pd.DataFrame, user_id: int) -> dict:
    """
    Extract all unique article_id values from the DataFrame that are not
    associated with the given user_id.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns user_id, article_id
    user_id (int): The user ID for which to exclude the articles.

    Returns:
    dict: A dictionary with the key as user_id and
    value as a list of unique article_id elements.
    """
    # Identify all article_id associated with the given user_id
    user_articles = df[df["user_id"] == user_id]["article_id"].to_list()

    # Identity users which are not user_id
    inverse_user = df.loc[df["user_id"] != user_id]

    # Extract all articles not clicked by user_id
    final_df = inverse_user[~inverse_user["article_id"].isin(user_articles)]

    # List of articles without duplicates
    article_ids = final_df["article_id"].unique().tolist()

    result = {
        "user_id": user_id,
        "article_id": article_ids,
    }

    return result
