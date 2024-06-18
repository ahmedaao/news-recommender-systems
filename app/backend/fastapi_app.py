import os
import pickle
from fastapi import FastAPI
from dotenv import load_dotenv
from src import dataset
from src.modeling import predict

# Load environment variables from .env file
load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")

app = FastAPI(title="MyApp", description="News Recommender System")


@app.post("/closest_articles")
def cbf(article_id: int, nb_closest_articles: int):
    result = dataset.closest_articles(
        os.path.join(ROOT_DIR, "models", "articles_embeddings.pickle"),
        article_id,
        nb_closest_articles
    )

    return result


# Load dataset manually
file_path = os.path.join(ROOT_DIR, "app", "backend", "dataset.pickle")
with open(file_path, "rb") as file:
    df = pickle.load(file)

@app.post("/recommended_articles")
def cf_algo_svd(user_id: int, nb_articles: int):
    result = predict.cf_svd(ROOT_DIR, user_id, df, nb_articles)

    return result
