"""Module containing function to make prediction"""

import os
import pickle
import pandas as pd
from src import dataset


def cf_baseline_only(
    ROOT_DIR: str, user_id: int, df: pd.DataFrame, nb_articles_to_print: int
):
    # Deserialization pickle object
    with open(
        os.path.join(ROOT_DIR, "models/model_baseline_only.pickle"), "rb"
    ) as file:
        best_model_baseline_only = pickle.load(file)

    # List all articles not click by a specific user
    articles_not_clicked_by_user = dataset.articles_not_clicked_by_user(df, user_id)

    # Make predictions
    predictions = []
    for article_id in articles_not_clicked_by_user["article_id"]:
        prediction = best_model_baseline_only.predict(
            uid=articles_not_clicked_by_user["user_id"], iid=article_id
        )
        predictions.append(
            {
                "user_id": prediction.uid,
                "article_id": prediction.iid,
                "predicted_rating": prediction.est,
                "details": prediction.details,
            }
        )

    # Sort the predictions in descending order by 'predicted_rating'
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = predictions[:nb_articles_to_print]

    article_ids = [entry["article_id"] for entry in predictions]
    predicted_ratings = [entry["predicted_rating"] for entry in predictions]

    result = {
        "article_ids": article_ids,
        "predicted_ratings": predicted_ratings,
    }

    return result


def cf_knn(ROOT_DIR: str, user_id: int, df: pd.DataFrame, nb_articles_to_print: int):
    # Deserialization pickle object
    with open(os.path.join(ROOT_DIR, "models/model_based_knn.pickle"), "rb") as file:
        best_model_knn = pickle.load(file)

    # List all articles not click by a specific user
    articles_not_clicked_by_user = dataset.articles_not_clicked_by_user(df, user_id)

    # Make predictions
    predictions = []
    for article_id in articles_not_clicked_by_user["article_id"]:
        prediction = best_model_knn.predict(
            uid=articles_not_clicked_by_user["user_id"], iid=article_id
        )
        predictions.append(
            {
                "user_id": prediction.uid,
                "article_id": prediction.iid,
                "predicted_rating": prediction.est,
                "details": prediction.details,
            }
        )

    # Sort the predictions in descending order by 'predicted_rating'
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = predictions[:nb_articles_to_print]

    article_ids = [entry["article_id"] for entry in predictions]
    predicted_ratings = [entry["predicted_rating"] for entry in predictions]

    result = {
        "article_ids": article_ids,
        "predicted_ratings": predicted_ratings,
    }

    return result


def cf_svd(ROOT_DIR: str, user_id: int, df: pd.DataFrame, nb_articles_to_print: int):
    # Deserialization pickle object
    with open(os.path.join(ROOT_DIR, "models/model_based_svd.pickle"), "rb") as file:
        best_model_knn = pickle.load(file)

    # List all articles not click by a specific user
    articles_not_clicked_by_user = dataset.articles_not_clicked_by_user(df, user_id)

    # Make predictions
    predictions = []
    for article_id in articles_not_clicked_by_user["article_id"]:
        prediction = best_model_knn.predict(
            uid=articles_not_clicked_by_user["user_id"], iid=article_id
        )
        predictions.append(
            {
                "user_id": prediction.uid,
                "article_id": prediction.iid,
                "predicted_rating": prediction.est,
                "details": prediction.details,
            }
        )

    # Sort the predictions in descending order by 'predicted_rating'
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    predictions = predictions[:nb_articles_to_print]

    article_ids = [entry["article_id"] for entry in predictions]
    predicted_ratings = [entry["predicted_rating"] for entry in predictions]

    result = {
        "article_ids": article_ids,
        "predicted_ratings": predicted_ratings,
    }

    return result
