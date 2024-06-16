import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

data = pd.read_csv('spam.csv', encoding='latin1')
data.head()

data['v1'].value_counts()

data.columns

data.isnull().sum()

data.shape

le = LabelEncoder()
data['v1'] = le.fit_transform(data['v1'])
data = data[['v1','v2']]

data.head()

data['v1'].value_counts()

sns.countplot(data ,x='v1' ,hue='v1')
plt.show()

nltk.download('stopwords')

stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

corpus =[]
for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['v2'][i])
    review =review.lower()
    review = review.split()

    review = [stemmer.stem(words) for words in review if not words in stopwords]
    review = ' '.join(review)
    corpus.append(review)

from wordcloud import WordCloud
plt.figure(figsize=(14,8))
wordcloud=WordCloud(width=600,height=400, contour_color='black').generate(' '.join(corpus))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('CORPUS',fontsize=20)
plt.show()

vocabulary_size = 5000

one_hot_rep = [one_hot(words,vocabulary_size)for words in corpus]
one_hot_rep[1]

sent_length = 20
embedded_docs = pad_sequences(one_hot_rep,padding='post',maxlen=sent_length)
embedded_docs[10]

embedded_vector_length = 40
model = Sequential()
model.add(Embedding(vocabulary_size,embedded_vector_length,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics =['accuracy'])
model.summary()

y = data['v1']
y

X_final = np.array(embedded_docs)
y_final = np.array(y)

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X_final, y_final)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

Model= model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=32)

plt.plot(Model.history['accuracy'])
plt.plot(Model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['accuracy','val_accuracy'],loc='center')
plt.show()

plt.plot(Model.history['loss'])
plt.plot(Model.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Loss','val_loss'],loc='center')
plt.show()

prediction = model.predict(X_test)

predictions = np.where(prediction > 0.5 ,1,0)
predictions

conf =confusion_matrix(y_test,predictions)
conf

sns.heatmap(conf,annot = True, fmt='d',cmap='viridis')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

accuracy_score(y_test,predictions)

print(classification_report(y_test,predictions))

model.save('spam_detection.h5')
