# Import packages
import pandas as pd


def rating_implicite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate implicit ratings for articles based on user clicks and session size,
    and normalize the ratings to be between 0 and 10.

    This function processes a DataFrame to compute an implicit rating for each article
    by multiplying the number of clicks per article by the session size. The ratings are
    then normalized to fall between 0 and 10. The resulting DataFrame contains only the
    user ID, article ID, and the calculated normalized implicit rating.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing user interactions with articles.

    Returns:
    pd.DataFrame: DataFrame containing user ID, article ID, and the normalized implicit rating.
    """
    # Calculate the number of clicks per article
    clicks_per_article = df.groupby("article_id")["article_id"].transform("count")

    # Create implicit ratings, calculated as clicks per article times session size
    df["rating"] = clicks_per_article * df["session_size"]

    # Normalize the click_per_article column to be between 0 and 10
    min_clicks = df["rating"].min()
    max_clicks = df["rating"].max()
    df["rating"] = 10 * (df["rating"] - min_clicks) / (max_clicks - min_clicks)

    # Downsize dataframe columns
    df["user_id"] = df["user_id"].astype("uint32")
    df["article_id"] = df["article_id"].astype("uint32")
    df["rating"] = df["rating"].round(2)

    # Ascending column 'click_per_article'
    df = df.sort_values(by="rating", ascending=False)

    # Select only the relevant columns: user ID, article ID, and the calculated normalized implicit rating
    df = df[["user_id", "article_id", "rating"]]

    return df
