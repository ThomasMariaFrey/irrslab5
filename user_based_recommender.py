import pandas as pd
import numpy as np
from utils import load_dataset_from_source, split_users
from similarity import compute_similarity_matrix


def generate_m(ratings):
    # Create the user-item matrix where each row corresponds to a user and each column to a movie.
    M = pd.pivot_table(ratings, index='userId', columns='movieId', values='rating')
    return M


def user_based_recommender(target_user, M, users_similarity_matrix, k=10):
    # Find the k most similar users to the target user
    similarities = users_similarity_matrix[target_user]
    most_similar_users = similarities.nlargest(k + 1).iloc[1:]  # excluding self-comparison

    # Predict the interest for the target user based on similar users
    recommendations = {}
    for movie in M.columns:
        if np.isnan(M.loc[target_user, movie]):  # only predict for movies the target user hasn't rated
            weighted_sum = 0
            sim_sum = 0
            for similar_user in most_similar_users.index:
                if not np.isnan(M.loc[similar_user, movie]):
                    weighted_sum += most_similar_users[similar_user] * (
                                M.loc[similar_user, movie] - M.loc[similar_user].mean())
                    sim_sum += most_similar_users[similar_user]

            if sim_sum != 0:
                predicted_rating = M.loc[target_user].mean() + (weighted_sum / sim_sum)
                recommendations[movie] = predicted_rating

    # Sort the movies based on the predicted rating
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Return the top k recommended movies
    return sorted_recommendations[:k]


if __name__ == "__main__":
    path_to_data = 'ml-latest-small'
    dataset = load_dataset_from_source(path_to_data)

    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    M = generate_m(ratings)

    # Assume that split_users and compute_similarity functions are defined in utils.py and similarity.py respectively
    train, test = split_users(ratings)
    users_similarity_matrix = compute_similarity_matrix(train)

    # Choose a target user for demonstration (this should be part of the validation process)
    target_user = 1
    top_k_recommendations = user_based_recommender(target_user, M, users_similarity_matrix)
    print(top_k_recommendations)
