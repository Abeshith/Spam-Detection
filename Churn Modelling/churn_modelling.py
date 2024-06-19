# -*- coding: utf-8 -*-
"""Churn_Modelling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16UCC9EctwrkR0Hw7b--THNM6ar95x4QQ
"""

import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Churn_Modelling.csv')
df = data.copy()

data.head()

data.describe().T

buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
print(s)

data.isnull().any()

data['Geography'].value_counts()

le = LabelEncoder()
data['Geography'] = le.fit_transform(data['Geography'])
data['Gender'] = le.fit_transform(data['Gender'])

data = data.drop(['Surname'],axis=1)

data.head()

data.corr().T

data['Exited'].value_counts()

"""# **EXPLORATORY DATA ANALYSIS**"""

sns.countplot(data=df , x='Exited' , hue='Geography',palette='Blues')

churn = df.loc[df['Exited'] == 1]["Geography"].value_counts()
fig,axes = plt.subplots(figsize=(12,6))
sns.barplot(x=churn,y = churn.index ,palette='husl')

churn = df.loc[df['Exited'] == 0]["Geography"].value_counts()
fig,axes = plt.subplots(figsize=(12,6))
sns.barplot(x=churn,y = churn.index ,palette='husl')

unique_counts = data.nunique()
threshold = 12
continuous_vars = unique_counts[unique_counts > threshold].index.tolist()
if 'id' in continuous_vars:
    continuous_vars.remove('id')


target_column = 'Exited'

for column in continuous_vars:
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    # Plot histogram with hue
    sns.histplot(data=data, x=column, hue=target_column, bins=50, kde=True, ax=axes[0], palette='muted')
    axes[0].set_title(f'Histogram of {column} with {target_column} Hue')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Count')
    axes[0].legend(title=target_column, loc='upper right')

    # Plot KDE plot with hue
    sns.kdeplot(data=data, x=column, hue=target_column, ax=axes[1], palette='muted')
    axes[1].set_title(f'KDE Plot of {column} with {target_column} Hue')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Density')
    axes[1].legend(title=target_column, loc='upper right')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

churn_data = data.loc[:len(data) // 2]
churn_data = churn_data[['CreditScore','Age','Balance', 'EstimatedSalary','Exited']]
sns.pairplot(churn_data, hue='Exited', palette='husl', diag_kind='kde')

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(),annot=True)

X = data.drop(['Exited'],axis=1)
y = data['Exited']

"""# Handling imbalance data"""

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

"""# **CLASSIFIER MODEL**

# Logistic Regression
"""

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

parameters = {
    'lr__C': [0.1, 1, 10],
    'lr__solver': ['liblinear', 'saga'],
    'lr__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

test_score = grid_search.score(X_test, y_test)
print("Test set accuracy with best parameters:", test_score)

best_lr = grid_search.best_estimator_
best_lr

predictions = best_lr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
accuracy

print(classification_report(y_test, predictions))

"""# K Nearest Neighbour"""

error_rate = []
for i in range(1,40):
    KNN = KNeighborsClassifier(n_neighbors=i).fit(X_train,y_train)
    pred = KNN.predict(X_test)
    error_rate.append(np.mean(pred !=y_test))

plt.figure(figsize=(12,5))
plt.plot(range(1,40),error_rate,color='blue',markersize=10,markerfacecolor='red',linestyle='dashed',marker='o')

KNN = KNeighborsClassifier(n_neighbors=2,weights='uniform')
KNN.fit(X_train,y_train)

KNN_prediction = KNN.predict(X_test)

KNN_accuracy = accuracy_score(y_test,KNN_prediction)
KNN_accuracy

print(classification_report(y_test, KNN_prediction))

"""# Decision Tree Classifier"""

pipeline_dt = Pipeline([
    ('dt', DecisionTreeClassifier(random_state=42))
])

parameters_dt = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__splitter': ['best', 'random'],
    'dt__max_depth': [None, 10, 20, 30],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 4],
}

grid_search_dt = GridSearchCV(pipeline_dt, parameters_dt, cv=5, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

print("Best parameters found for Decision Tree:", grid_search_dt.best_params_)
print("Best cross-validation score for Decision Tree:", grid_search_dt.best_score_)

test_score_dt = grid_search_dt.score(X_test, y_test)
print("Test set accuracy with best parameters for Decision Tree:", test_score_dt)

best_dt = grid_search_dt.best_estimator_
predictions_dt = best_dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, predictions_dt)
accuracy_dt

print(classification_report(y_test, predictions_dt))

"""# Support Vector Machine"""

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42,kernel='linear',C= 1e6))
])

pipeline_svm.fit(X_train, y_train)

best_svm = pipeline_svm

predictions_svm = best_svm.predict(X_test)

accuracy_svm = accuracy_score(y_test, predictions_svm)
accuracy_svm

