import pandas as pd 
import utils as ut

def naive_recommender(ratings: object, movies:object, k: int = 10) -> list: 
    # Provide the code for the naive recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.
    # Aggregate the ratings to compute the average rating for each movie
    avg_ratings = ratings.groupby('movieId')['rating'].mean()

    # Count the number of ratings to find the most viewed movies
    count_ratings = ratings.groupby('movieId')['rating'].count()

    # Combine the average ratings with the count of ratings
    movie_stats = pd.DataFrame({
        'average_rating': avg_ratings,
        'number_of_ratings': count_ratings
    })

    # Join movie stats with the movies DataFrame to get the movie titles
    movie_stats = movie_stats.join(movies.set_index('movieId'))

    # Sort movies by average rating and number of ratings
    most_seen_movies = movie_stats.sort_values(by=['average_rating', 'number_of_ratings'], ascending=False)

    # Return the top k movies
    return most_seen_movies.head(k)


if __name__ == "__main__":
    
    path_to_ml_latest_small = 'ml-latest-small'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    print(naive_recommender(ratings, movies))

'''
The time complexity of the naive recommender system can be estimated based on the operations performed on the data:

Loading the data: This operation is typically O(n)O(n), where nn is the number of entries in the file.
Aggregating and sorting the data: This is usually O(nlg⁡n)O(nlogn) because sorting is involved.
Slicing the top kk entries: This operation is O(k)O(k), which is constant time for a fixed kk and negligible compared to the sorting operation.
Thus, the overall time complexity is dominated by the sorting operation, which is O(nlng⁡n)O(nlogn).

For the limitation, a naive recommender system like this does not account for the personalized preferences of individual users. It recommends the same movies to everyone based on overall popularity, which might not be relevant to all users. Additionally, it doesn't handle the cold start problem well (i.e., recommending items to new users with no history).
'''