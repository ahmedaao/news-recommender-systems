# Import packages
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src import dataset
from src.modeling import predict

# Load environment variables from .env file
load_dotenv()

# Load pickle object
df = dataset.load_pickle_file(
    os.path.join(os.getenv("ROOT_DIR"), "app", "backend", "dataset.pickle")
)

app = FastAPI(title="MyApp", description="News Recommender System")


class RecommendationRequest(BaseModel):
    selected_user_id: int
    random_article_id: int
    nb_articles: int


@app.post("/content_based_filtering")
def cbf(request: RecommendationRequest):
    article_id = request.random_article_id
    nb_articles = request.nb_articles

    result = dataset.closest_articles(
        os.path.join(os.getenv("ROOT_DIR"), "models", "articles_embeddings.pickle"),
        article_id,
        nb_articles
    )

    return result


@app.post("/collaborative_filtering_knnWithMeans")
def cf_algo_knn(request: RecommendationRequest):
    user_id = request.selected_user_id
    nb_articles = request.nb_articles

    result = predict.cf_knn(os.getenv("ROOT_DIR"), user_id, df, nb_articles)

    return result


@app.post("/collaborative_filtering_svd")
def cf_algo_svd(request: RecommendationRequest):
    user_id = request.selected_user_id
    nb_articles = request.nb_articles

    result = predict.cf_svd(os.getenv("ROOT_DIR"), user_id, df, nb_articles)

    return result
