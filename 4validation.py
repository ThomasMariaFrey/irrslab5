import pandas as pd
from utils import matrix_genres, split_users, load_dataset_from_source
from similarity import compute_similarity, compute_similarity_matrix
from naive_recommender import naive_recommender
import user_based_recommender

def validate_recommender(recommendations, user_validation_set, matrix_genres):
    # Calculate the frequency of each genre in the user's validation set
    user_genres = user_validation_set.merge(matrix_genres, on='movieId')
    true_genre_distribution = user_genres.drop(columns=['userId', 'movieId', 'rating']).mean()

    # Calculate the frequency of each genre in the recommended movies
    recommended_genres = recommendations.merge(matrix_genres, on='movieId')
    recommended_genre_distribution = recommended_genres.drop(columns=['movieId']).mean()

    # Evaluate the recommender based on how similar the recommended genre distribution is to the true genre distribution
    # This could be done using cosine similarity, Pearson correlation, or other statistical measures.
    similarity = compute_similarity(true_genre_distribution, recommended_genre_distribution)
    return similarity

if __name__ == "__main__":
    path_to_data = 'ml-latest-small'
    dataset = load_dataset_from_source(path_to_data)

    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    train_set, validation_set = split_users(ratings)

    M = user_based_recommender.generate_m(ratings)

    # Assume that split_users and compute_similarity functions are defined in utils.py and similarity.py respectively
    train, test = split_users(ratings)
    users_similarity_matrix = user_based_recommender.compute_similarity_matrix(train)

    # Assume rec1 and rec2 are functions that return top k recommendations for a given user
    user_id = 1  # example user ID
    top_k_rec1 = naive_recommender(ratings, movies)
    top_k_rec2 = user_based_recommender.user_based_recommender(user_id,M,users_similarity_matrix=users_similarity_matrix, k=10)

    # Assume matrix_genres is a DataFrame that maps movie IDs to genres
    matrix_genres = matrix_genres(M)

    # Get the actual movies rated by the user in the validation set
    user_validation_set = validation_set[validation_set['userId'] == user_id]

    # Validate each recommender
    similarity_rec1 = validate_recommender(top_k_rec1, user_validation_set, matrix_genres)
    similarity_rec2 = validate_recommender(top_k_rec2, user_validation_set, matrix_genres)

    print(f"Similarity score for Rec1: {similarity_rec1}")
    print(f"Similarity score for Rec2: {similarity_rec2}")
