import pandas as pd
import streamlit as st
import requests
import pickle


def recommend(name):
   index = movies_df[movies_df['title'] == name].index[0]
   distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
   recommended_movies = []
   for i in distances:
       # print(i)
       recommended_movies.append(movies_df.iloc[i[0]]['title'])
   return recommended_movies


similarity = pickle.load(open("similarity.pkl",'rb'))
movies = pickle.load(open("movies.pkl",'rb'))

st.title("Content Based Recommendation of Movies")

movie_list = movies['title'].values()
list_movies =[]
for i in movie_list:
    list_movies.append(i)
movies_df = pd.DataFrame(movies)

movie_selected = st.selectbox("Select the movie name here: ", list_movies)


if st.button('Show Recommendation'):
    recommended_movies = recommend(movie_selected)
    st.subheader("Selected movie: {}".format(movie_selected))
    count = 1
    st.subheader("Recommendations from engine")
    for i in recommended_movies:
        st.markdown("{}.  {}".format(count,i))
        count+=1



