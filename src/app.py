import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from recommender import MovieRecommender

# Initialize recommender (cached to avoid reloading every time)
@st.cache_resource
def load_recommender():
    return MovieRecommender("data/ratings.csv", "data/movies.csv")

recommender = load_recommender()

# Streamlit UI
st.title("🎬 Movie Recommendation System (User-User k-NN)")

# User input
user_id_input = st.number_input(
    "Enter User ID:",
    min_value=int(recommender.user_movie_matrix.index.min()),
    max_value=int(recommender.user_movie_matrix.index.max()),
    value=1
)
top_n_input = st.slider("Number of Recommendations:", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    recommended_movies = recommender.recommend_movies(user_id_input, top_n=top_n_input, k=5)

    if recommended_movies.empty:
        st.write(f"No recommendations available for User {user_id_input}.")
    else:
        st.write(f"Top {top_n_input} movie recommendations for User {user_id_input}:")

        # Display table
        rec_df = pd.DataFrame({
            'Movie': recommended_movies.index,
            'Predicted Rating': recommended_movies.values
        })
        st.dataframe(rec_df)

        # Plot chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(recommended_movies.index[::-1], recommended_movies.values[::-1], color='skyblue')
        ax.set_xlabel("Predicted Rating")
        ax.set_title(f"Top {top_n_input} Movie Recommendations")
        st.pyplot(fig)
