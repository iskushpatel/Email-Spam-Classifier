import streamlit as st
import joblib
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

try:
    mnb_model = joblib.load('mnb_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'mnb_model.joblib' and 'tfidf_vectorizer.joblib' are in the same directory.")
    st.stop()

st.title('Spam Message Classifier')

user_message = st.text_area("Enter a message:")

if st.button('Classify Message'):
    if user_message:
        cleaned_message = clean_text(user_message)
        transformed_message = tfidf_vectorizer.transform([cleaned_message])
        prediction = mnb_model.predict(transformed_message)

        if prediction[0] == 1:
            st.write("This message is: **Spam**")
        else:
            st.write("This message is: **Not Spam**")
    else:
        st.warning("Please enter a message to classify.")