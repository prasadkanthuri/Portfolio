import streamlit as st

import helper

import pickle

model = pickle.load(open("model.pkl",'rb'))


st.title("Quora Duplicate Question Pair Detector")

q1 = st.text_input("Enter Question1")

q2 = st.text_input("Enter Question2")


count = 0
if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]
    if count==0:
        st.text("Fetching the result...")
        count=1
    else:
        st.text("Fetching the result..")
        count=0



    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')
