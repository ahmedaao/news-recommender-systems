# Import packages
import pickle
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import accuracy


def model_based_knn(df: pd.DataFrame, model_filename: str):
    """
    Train a collaborative filtering model using k-Nearest Neighbors
    with Means and save the best model.

    Parameters:
    df (pd.DataFrame): DataFrame containing the user-item ratings.
    It should have columns corresponding to user IDs, item IDs, and ratings.
    model_filename (str): The filename to save the trained model as a pickle.

    Returns:
    best_model: The best trained k-NN model based on RMSE.
    """
    # Define the rating scale
    reader = Reader(rating_scale=(0, 10))

    # Loads Pandas Dataframe
    data = Dataset.load_from_df(df, reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Set similarity options
    sim_options = {
        "name": ["msd"],
        "min_support": [3],
        "user_based": [False],
    }

    param_grid = {"sim_options": sim_options}

    # Perform grid search with cross-validation
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
    gs.fit(data)

    # Get the best model from the grid search
    best_model = gs.best_estimator["rmse"]

    # Train the best model on the full training set
    best_model.fit(trainset)

    # Train the best model on the full training set
    # trainset = data.build_full_trainset()
    # best_model.fit(trainset)

    # Save the best model to a file using pickle
    with open(model_filename, "wb") as model_file:
        pickle.dump(best_model, model_file)

    # Make predictions on the test set
    predictions = best_model.test(testset)

    # Calculate the RMSE
    rmse = accuracy.rmse(predictions, verbose=True)

    return best_model, rmse
