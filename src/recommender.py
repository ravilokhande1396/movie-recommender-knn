import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, ratings_path: str, movies_path: str):
        # Load data
        ratings = pd.read_csv(ratings_path)  # userId,movieId,rating,timestamp
        movies = pd.read_csv(movies_path)    # movieId,title,genres

        # Merge ratings with movies
        data = ratings.merge(movies, on='movieId')

        # Aggregate duplicate ratings (mean)
        data_agg = data.groupby(['userId', 'title'])['rating'].mean().reset_index()

        # User-Movie Matrix
        self.user_movie_matrix = data_agg.pivot(index='userId', columns='title', values='rating').fillna(0)

        # User similarity (cosine)
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

    def recommend_movies(self, user_id: int, top_n: int = 10, k: int = 5):
        """Recommend top_n movies for a given user_id using k nearest neighbors."""

        # Find k most similar users
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[1:k+1].index

        # Weighted ratings
        weighted_ratings = self.user_movie_matrix.loc[similar_users].T.dot(
            self.user_similarity_df[user_id][similar_users]
        )
        sum_sim = self.user_similarity_df[user_id][similar_users].sum()
        if sum_sim == 0:
            sum_sim = 1e-8
        recommendation_scores = weighted_ratings / sum_sim

        # Remove already-rated movies
        user_rated = self.user_movie_matrix.loc[user_id]
        recommendation_scores = recommendation_scores[user_rated == 0]

        # Top N recommendations
        recommended = recommendation_scores.sort_values(ascending=False).head(top_n)
        return recommended
