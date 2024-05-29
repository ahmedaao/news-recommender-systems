# Import packages
import numpy as np
import _pickle as cPickle
from sklearn.metrics.pairwise import cosine_similarity


def closest_articles(model_dir_path: str,
                     article_id: int,
                     nb_closest_articles: int) -> dict:
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
    with open(model_dir_path + 'articles_embeddings.pickle', 'rb') as f:
        articles_embeddings = cPickle.load(f)

    row = articles_embeddings[article_id].reshape(1, -1)
    cosine_similarities = cosine_similarity(row, articles_embeddings).flatten()

    sorted_indices = np.argsort(-cosine_similarities)[1:][:nb_closest_articles]
    sorted_cosine_similarities = np.sort(cosine_similarities)[::-1][1:][:nb_closest_articles]

    results = {
        "indices": sorted_indices,
        "cosine_similarities": sorted_cosine_similarities
    }

    return results
