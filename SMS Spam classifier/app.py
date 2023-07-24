import streamlit as st


import numpy as np

import pickle
cv = pickle.load(open('cv.pkl','rb'))
RF = pickle.load(open('RF.pkl','rb'))
# vectorize_input = pickle.load(open("vectorize_input.pkl",'rb'))

import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
ps = PorterStemmer()

def preprocess(sms):
    # Converting the text to lower case
    sms = sms.lower()
    sms = word_tokenize(sms)
    # Keeping only the alphanumeric part of the text
    y = []
    for i in sms:
        if i.isalnum():
            y.append(i)
    preprocessed_sms = y[:]
    y.clear()

    # Performing stemming after removing the stopwords
    for i in preprocessed_sms:
        if i not in stop_words and i not in string.punctuation:
            y.append(ps.stem(i))
    preprocessed_sms = y[:]
    y.clear()
    return " ".join(preprocessed_sms)


def vectorize_input(text):
    # feature extraction
    words_count = len(text.split(" "))
    sentences_count = len(sent_tokenize(text))
    charecters_count = len(text)
    additional_features = np.array([words_count, sentences_count, charecters_count])

    print(additional_features)
    # Preprocesing the data
    pre_processed_text = preprocess(text)
    # print([pre_processed_text])

    # Vectorizing the text
    vectorized_text = cv.transform([pre_processed_text]).toarray()
    # print(additional_features)
    vectorized_text = vectorized_text.reshape(-1)
    vectorized_input = np.hstack((additional_features, vectorized_text))
    return vectorized_input



st.title("SMS SPAM Classifier")

sms = st.text_input("Enter the message here")

if st.button("Check now"):
    text = vectorize_input(sms)
    result = RF.predict(text.reshape(1, -1))
    if result[0] ==0:
        st.header("Ham")
    else:
        st.header("Spam")





