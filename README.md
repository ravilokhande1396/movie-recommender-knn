A simple Movie Recommender System built using User-User k-Nearest Neighbors (k-NN) with the  movie dataset (https://grouplens.org/datasets/movielens/)

This project demonstrates how collaborative filtering can be applied to recommend movies to users based on similar user preferences.
movie-recommender-knn/
Project structure: The User should place all files into the respective folder:

│── data/

     1. movies.csv
     
     2.ratings.csv
│── src/

     1.recommender.py    # core logic
     
     2. app.py           # Streamlit UI
     
│── requirements.txt

Install all dependencies:

pip install -r requirements.txt

Then run this command:

streamlit run src/app.py, so the link will open in a web browser like  (http://localhost:8501) 

So the user can access recommended movies by other users 

