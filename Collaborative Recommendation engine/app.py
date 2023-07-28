import pandas as pd
import streamlit as st

import numpy as np
import pickle

books_list = pd.read_pickle("books_list.pkl")
books_name = books_list.index

similarity_scores = pd.read_pickle("similarity_scores.pkl")

books_info= pd.read_pickle("books_info.pkl")



st.title('Recommendation System')
st.header("""Collaborative Recommendation Engine:
It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input."""
          )

movie_selected = st.selectbox("Search for the book here",[i for i in books_name])

def recommend(book_name):
    books_result_name = []
    books_result_poster = []
    index = np.where(books_list.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[:5]
    for i in similar_books:
        similar_book_name = books_list.index[i[0]]
        book_name = similar_book_name
        url = books_info[books_info['Book-Title']==book_name]['Image-URL-M'].reset_index()['Image-URL-M'][0]
        books_result_poster.append(url)
        books_result_name.append(book_name)
    return books_result_poster,books_result_name


if st.button("Search"):
    books_result_poster,books_result_name = recommend(movie_selected)
    col1, col2,col3 = st.columns(3)
    with col1:
        st.text(books_result_name[0])
        st.image(books_result_poster[0])
    with col2:
        st.text(books_result_name[1])
        st.image(books_result_poster[1])

    with col3:
        st.text(books_result_name[2])
        st.image(books_result_poster[2])

    col1, col2,col3 = st.columns(3)

    with col1:
        st.text(books_result_name[3])
        st.image(books_result_poster[3])
    with col2:
        st.text(books_result_name[4])
        st.image(books_result_poster[4])
