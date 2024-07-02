# Afame-Technologies

# Spam Classification Using Deep Learning

This project demonstrates a deep learning approach to spam classification using Python, TensorFlow, and Natural Language Processing (NLP) techniques. The model is built with a Bidirectional LSTM to classify messages as spam or ham.

## Dataset
The dataset used for this project is from a CSV file named `spam.csv`, which contains the following columns:
- `v1`: Label (ham or spam)
- `v2`: Message content

## Project Steps
1. **Importing Libraries**: Essential libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, and TensorFlow are imported.
2. **Data Preprocessing**:
    - Load the dataset using `pandas`.
    - Encode labels using `LabelEncoder`.
    - Remove unnecessary columns and handle missing values.
    - Visualize the distribution of spam and ham messages using Seaborn.
3. **Text Preprocessing**:
    - Clean the text data by removing non-alphabetic characters and converting to lowercase.
    - Remove stopwords and apply stemming using NLTK.
4. **Word Cloud**: Generate a word cloud to visualize the most common words in the corpus.
5. **Text Encoding**:
    - Convert the text data to one-hot representations.
    - Pad sequences to ensure uniform input length.
6. **Model Building**:
    - Use TensorFlow to build a Sequential model.
    - Add an Embedding layer followed by a Bidirectional LSTM layer.
    - Add a Dense layer with a sigmoid activation function.
7. **Handling Class Imbalance**:
    - Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
8. **Train-Test Split**: Split the data into training and testing sets.
9. **Model Training**:
    - Train the model with the training data.
    - Visualize the accuracy and loss over epochs.
10. **Evaluation**:
    - Evaluate the model using accuracy score, confusion matrix, and classification report.
    - Save the trained model to a file `spam_detection.h5`.

## **Streamlit Application**

I have also developed a Streamlit application to classify sentences using the trained model (spam_detection.h5). The application provides an easy-to-use interface for users to input a sentence and get a classification result (spam or ham).

<img src="https://github.com/Abeshith/Computer-Vision/blob/main/output%20images/Detect-FromPhotos.png" width="400">

