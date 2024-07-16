# Import packages
import os
import json
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src import dataset

# Load environment variables from .env file
load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")

# Load pickle object
df = dataset.load_pickle_file(
    os.path.join(ROOT_DIR, "app", "backend", "dataset.pickle")
)


def app():
    st.title("News Recommender System")

    # Summary of the app functionality
    st.write(
        """This basic application is a MVP (Minimal Viable Product) which take
        a user id as input and generate a list of the n most interesting
        articles for this user thanks to a Deep Learning model embedded
        into a API"""
    )

    st.sidebar.write("## Input Data :gear:")

    # Extract unique user_id, article_id and nb_articles
    unique_user_ids = df["user_id"].unique().tolist()
    selected_user_id = st.sidebar.selectbox("Choose an user_id", unique_user_ids)
    st.write(f"You selected user_id: {selected_user_id}")
    if selected_user_id:
        user_articles = df[df["user_id"] == selected_user_id]["article_id"]
    if not user_articles.empty:
        random_article_id = user_articles.sample(1).iloc[0]
        st.write(
            f"Random article_id associated: {random_article_id}\n"
            "(Only useful for the content-based filtering model)"
        )
    else:
        st.write(f"No articles found for user_id {selected_user_id}")

    nb_articles = st.sidebar.number_input(
        "Number of articles to recommend",
        min_value=5,
        value=5
    )

    result = {
        "selected_user_id": int(selected_user_id),
        "random_article_id": int(random_article_id),
        "nb_articles": int(nb_articles)
    }

    # Serialize and send inference request to fastAPI
    endpoint = st.sidebar.selectbox(
        "Choose the API endpoint",
        [
            "content_based_filtering",
            "collaborative_filtering_knnWithMeans",
            "collaborative_filtering_svd"
        ]
    )

    # When running containers separatly with simple docker
    # url = f"http://0.0.0.0:8000/{endpoint}"
    url = f"http://127.0.0.1:8000/{endpoint}"

    # When running containers through docker-compose to combine them
    # url = f"http://backend:8000/{endpoint}"

    headers = {
        "Content-Type": "application/json"
    }

    result_json = json.dumps(result)
    response = requests.post(
        url,
        headers=headers,
        data=result_json,
        timeout=120
    )

    # Retrieve result from fastAPI and print it
    if response.status_code == 200:
        recommendations = response.json()
        st.write("Recommended Articles:", pd.DataFrame(recommendations))
    else:
        st.write("Error:", response.status_code)


if __name__ == "__main__":
    app()
