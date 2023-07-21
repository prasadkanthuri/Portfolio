import pandas as pd
import numpy as np
import re
import pickle
from fuzzywuzzy import fuzz


import nltk



stop_words= pickle.load(open('stop_words.pkl','rb'))


cv = pickle.load(open('vectorizer.pkl','rb'))

def preprocessing_the_text(q):
    q = q.lower().strip()

    # Replacing some of the special charecters with their meaning
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # Expanding the short form of the words
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    expanded_words = []
    for i in q.split():
        if i in contractions:
            j = contractions[i]
            print(j)
        else:
            j = i
        expanded_words.append(j)
    q = " ".join(expanded_words)

    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    pattern = r'<[^>]+>'
    q = re.sub(pattern, '', q)

    # Removing URL's

    pattern = r'(https?://|www\.)[^ ]+'
    q = re.sub(pattern, '', q)

    # Removing punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q)

    return q


def query_point_creator(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocessing_the_text(q1)
    q2 = preprocessing_the_text(q2)

    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    def common_words(q1, q2):

        q1_words = set(q1.split(" "))
        q2_words = set(q2.split(" "))
        common_words = q1_words.intersection(q2_words)
        return len(common_words)

    input_query.append(common_words(q1, q2))
    input_query.append(common_words(q1, q2) * 100 / len(q1.split(" ")))
    input_query.append(common_words(q1, q2) * 100 / len(q2.split(" ")))

    # first_word_eq, Last_word_eq, len_diff

    def first_word_eq(q1, q2):
        q1_word1 = q1.split(" ")[0]
        q2_word1 = q2.split(" ")[0]
        return int(q1_word1 == q2_word1)

    def last_word_eq(q1, q2):
        q1_word_last = q1.split(" ")[-1]
        q2_word_last = q2.split(" ")[-1]
        return int(q1_word_last == q2_word_last)

    first_word_eq = first_word_eq(q1, q2)
    input_query.append(first_word_eq)
    last_word_eq = last_word_eq(q1, q2)
    input_query.append(last_word_eq)
    len_diff = abs(len(q1.split(" ")) - len(q2.split(" ")))
    input_query.append(len_diff)

    # non_sopwords_common, stopwords_commom
    def non_stopwords_common(q1, q2):
        q1 = q1.split(" ")
        q2 = q2.split(" ")
        q1_non_stop_words = []
        q2_non_stop_words = []
        for i in q1:
            if i not in stop_words:
                q1_non_stop_words.append(i)
        for i in q2:
            if i not in stop_words:
                q2_non_stop_words.append(i)
        q1_set = set(q1_non_stop_words)
        q2_set = set(q2_non_stop_words)
        common = q1_set.intersection(q2_non_stop_words)
        return len(common)

    non_sopwords_common = non_stopwords_common(q1, q2)
    input_query.append(non_sopwords_common)
    stopwords_commom = common_words(q1, q2) - non_sopwords_common
    input_query.append(stopwords_commom)

    # fuzz_features

    def test_fetch_fuzzy_features(q1, q2):

        fuzzy_features = [0.0] * 4

        # fuzz_ratio
        fuzzy_features[0] = fuzz.QRatio(q1, q2)

        # fuzz_partial_ratio
        fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

        # token_sort_ratio
        fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

        # token_set_ratio
        fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

        return fuzzy_features

    fuzzy_features = test_fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    # avg_words_length
    input_query.append((len(q1.split(" ")) + len(q1.split(" "))) / 2)

    # bow feature for q1
    q1_bow = cv.transform([q1]).toarray()

    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 17), q1_bow, q2_bow))
