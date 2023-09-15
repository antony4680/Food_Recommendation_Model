import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re
import string

df = pd.read_csv('updated_food_data2.csv')

def text_cleaning(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df['describe'] = df['describe'].apply(text_cleaning)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['describe'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['food']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    food_indices = [i[0] for i in sim_scores]
    return df['food'].iloc[food_indices]

features = ['type', 'veg_non', 'describe']

def create_soup(x):
    return x['type'] + " " + x['veg_non'] + " " + x['describe']

df['soup'] = df.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['food'])

rating = pd.read_csv('latest_rating.csv')

food_rating = rating.groupby(by='Food_ID').count()
food_rating = food_rating['Rating'].reset_index().rename(columns={'Rating': 'Rating_count'})

rating_matrix = rating.pivot_table(index='Food_ID', columns='user_id', values='Rating').fillna(0)

csr_rating_matrix = csr_matrix(rating_matrix.values)

recommender = NearestNeighbors(metric='cosine')
recommender.fit(csr_rating_matrix)

# Save the trained objects
with open('tfidf.pkl', 'wb') as file:
    pickle.dump(tfidf, file)
with open('count.pkl', 'wb') as file:
    pickle.dump(count, file)
with open('recommender.pkl', 'wb') as file:
    pickle.dump(recommender, file)

# Save the matrices
with open('tfidf_matrix.pkl', 'wb') as file:
    pickle.dump(tfidf_matrix, file)
with open('count_matrix.pkl', 'wb') as file:
    pickle.dump(count_matrix, file)
with open('rating_matrix.pkl', 'wb') as file:
    pickle.dump(rating_matrix, file)