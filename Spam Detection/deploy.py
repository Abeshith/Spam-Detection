import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

vocabulary_size = 5000
sent_length = 20
embedded_vector_length = 40

model = Sequential()
model.add(Embedding(vocabulary_size, embedded_vector_length, input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the weights into the model
try:
    model.load_weights('spam_detection.h5')
except Exception as e:
    st.error(f"Error loading model weights: {e}")

# Function to predict spam or not spam
def predict_spam(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Tokenize and pad sequences
    text_sequence = [one_hot(processed_text, vocabulary_size)]
    padded_sequence = pad_sequences(text_sequence, padding='post', maxlen=sent_length)
    # Make prediction
    prediction = model.predict(padded_sequence)
    # Return the predicted class (0 or 1)
    return int(prediction[0][0] > 0.5)

# Streamlit application
def main():
    st.title("Spam Detection Application")
    st.write("Enter a message to check if it is spam or not spam.")

    # Input text box for user to input message
    user_input = st.text_area("Message")

    if st.button("Predict"):
        if user_input:
            prediction = predict_spam(user_input)
            if prediction == 0:
                st.error("Spam")
            else:
                st.success("Not Spam")
        else:
            st.warning("Please enter a message to predict.")

if __name__ == '__main__':
    main()
