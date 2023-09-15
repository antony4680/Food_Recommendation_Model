import pickle
import pandas as pd
import numpy as np

# Load the trained objects
with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)
with open('count.pkl', 'rb') as file:
    count = pickle.load(file)
with open('recommender.pkl', 'rb') as file:
    recommender = pickle.load(file)

# Load the matrices
with open('tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)
with open('count_matrix.pkl', 'rb') as file:
    count_matrix = pickle.load(file)
with open('rating_matrix.pkl', 'rb') as file:
    rating_matrix = pickle.load(file)

def Get_Recommendations(food_id_list):
    recommended_food_ids = set()  # Use a set instead of a list to store unique food IDs
    
    for idx, food_id in enumerate(food_id_list):
        try:
            user_index = np.where(rating_matrix.index == int(food_id))[0][0]
            user_ratings = rating_matrix.iloc[user_index]

            reshaped = user_ratings.values.reshape(1, -1)
            distances, indices = recommender.kneighbors(reshaped, n_neighbors=10)

            nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]

            recommended_food_ids_per_id = nearest_neighbors_indices.tolist()
            recommended_food_ids_per_id = [id for id in recommended_food_ids_per_id if id not in food_id_list]

            if idx == 0:
                recommended_food_ids.update(recommended_food_ids_per_id[:4])
            else:
                recommended_food_ids.update(recommended_food_ids_per_id[:3])

        except IndexError:
            print(f"Food ID {food_id} not found in rating matrix.")

    remaining_recommendations = 10 - len(recommended_food_ids)
    if remaining_recommendations > 0:
        reshaped = rating_matrix.iloc[user_index].values.reshape(1, -1)
        distances, indices = recommender.kneighbors(reshaped, n_neighbors=remaining_recommendations)

        nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]

        additional_recommendations = nearest_neighbors_indices.tolist()
        additional_recommendations = [id for id in additional_recommendations if id not in recommended_food_ids]

        recommended_food_ids.update(additional_recommendations[:remaining_recommendations])

    if len(recommended_food_ids) < 10:
        reshaped = rating_matrix.iloc[user_index].values.reshape(1, -1)
        distances, indices = recommender.kneighbors(reshaped, n_neighbors=10)

        nearest_neighbors_indices = rating_matrix.iloc[indices[0]].index[1:]

        additional_recommendations = nearest_neighbors_indices.tolist()
        additional_recommendations = [id for id in additional_recommendations if id not in recommended_food_ids]

        recommended_food_ids.update(additional_recommendations[:10 - len(recommended_food_ids)])

    return list(recommended_food_ids)[:10]  # Convert the set back to a list

