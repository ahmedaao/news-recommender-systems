"""Module containing functions creating visualizations"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def article_distribution_by_users(df: pd.DataFrame, ROOT_DIR: str):
    """
    Visualizes the distribution of article counts by user

    This function groups the data by `user_id`, counts the number of
    `article_id` for each user.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns `user_id` and `article_id`.

    Returns:
    None
    """
    # Group by user_id and count the number of article_id for each user
    grouped_by_user = df.groupby("user_id")["article_id"].agg(["count"]).reset_index()

    # Calculate deciles for the article counts
    deciles = grouped_by_user["count"].quantile(np.arange(0.1, 1.1, 0.1))

    # Plot the distribution of article counts with a histogram
    plt.figure(figsize=(10, 6))
    grouped_by_user["count"].plot(kind="hist", bins=100, edgecolor="black", alpha=0.7)

    # Set the y-axis to a logarithmic scale
    plt.yscale("log")
    plt.title("Distribution of Article Counts per User")
    plt.xlabel("article_count")
    plt.ylabel("Number of Users (Log Scale)")

    # Save the plot if save_path is provided
    if ROOT_DIR:
        plt.savefig(
            os.path.join(ROOT_DIR, "reports/figures/01.png"), bbox_inches="tight"
        )

    # Show the plot
    plt.show()

    return None
