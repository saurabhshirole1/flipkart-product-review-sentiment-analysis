# # Sentiment Analysis of Flipkart Product Reviews


# ### Step 2: Feature Extraction & Model Training


# ### 1. Import All Required Libraries


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report

import pickle

# ### Observation:
# All required libraries for data handling, feature extraction, model training,
# and evaluation were successfully imported.
# 


# ### Step 1: Load the Cleaned Dataset


df = pd.read_csv("data\cleaned_flipkart_reviews.csv")

X = df['clean_review']
y = df['sentiment']

# ### Observation:
# - The cleaned dataset was loaded successfully.
# - The feature variable consists of preprocessed review text, while the target
# variable represents binary sentiment labels.
# 


# ### Step 2: Split Data into Training and Testing Sets


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

# ### Observation:
# - The dataset was split into training and testing sets using an 80:20 ratio.
# - Stratified sampling was applied to preserve the original class distribution.
# 


# ### Step 3: Convert Text Data into Numerical Form (TF-IDF)


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ### Observation:
# - TF-IDF vectorization was used to transform textual data into numerical features.
# - The use of unigrams and bigrams helps capture meaningful word patterns present in reviews.
# 


# ### Step 4: Train Logistic Regression Model


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)

# ### Observation:
# - The Logistic Regression model was trained on TF-IDF features.
# - It is well-suited for high-dimensional sparse text data.
# 


# ### Step 5: Evaluate Logistic Regression Model


f1_lr = f1_score(y_test, y_pred_lr)
print("Logistic Regression F1-Score:", f1_lr)
print(classification_report(y_test, y_pred_lr))

# ### Observation:
# Logistic Regression achieved a strong F1-score, indicating effective performance
# in classifying positive and negative reviews.
# 


# ### Step 6: Train Multinomial Naive Bayes Model


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

# ### Observation:
# - The Naive Bayes model was trained as a comparative baseline model.
# - It is computationally efficient but makes strong independence assumptions.
# 


# ### Step 7: Evaluate Naive Bayes Model


f1_nb = f1_score(y_test, y_pred_nb)
print("Naive Bayes F1-Score:", f1_nb)

# ### Observation:
# Naive Bayes achieved a slightly lower F1-score compared to Logistic Regression,
# which is expected due to its simplified probabilistic assumptions.
# 


# ### Step 8: Compare Model Performance


comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression (TF-IDF)",
        "Naive Bayes (TF-IDF)"
    ],
    "F1-Score": [
        f1_lr,
        f1_nb
    ]
})

comparison_df


# ### Observation:
# Logistic Regression outperformed Naive Bayes based on the F1-score and was selected
# as the final model for deployment.


# ### Step 9: Save the Best Model and Vectorizer


with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# ### Observation:
# The trained model and TF-IDF vectorizer were saved for reuse in real-time
# sentiment prediction through a web application.
#