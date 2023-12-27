import numpy as np
import pandas as pd

def compute_similarity_matrix(ratings):
    # Create a user-item matrix
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

    # Fill NaN values with 0, as we will compute the cosine similarity which handles 0s
    user_item_matrix = user_item_matrix.fillna(0)

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(user_item_matrix)

    # Create a DataFrame from the numpy matrix and assign index and column names as user IDs
    user_ids = user_item_matrix.index
    similarity_df = pd.DataFrame(similarity_matrix, index=user_ids, columns=user_ids)

    return similarity_df


def cosine_similarity(matrix):
    # Normalize the matrix rows to unit (L2) norm
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / matrix_norm

    # Compute the cosine similarity as matrix multiplication of the normalized matrix with its transpose
    similarity = np.dot(matrix, matrix.T)

    # Ensure the diagonal is all 1s (as a user is perfectly similar to itself)
    np.fill_diagonal(similarity, 1)

    return similarity


if __name__ == "__main__":
    print("NOTHING")
    # Assuming 'train' is a pandas DataFrame with columns 'userId', 'movieId', and 'rating'
    #users_similarity_matrix = compute_similarity_matrix(train)