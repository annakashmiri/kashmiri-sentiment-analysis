# Kashmiri Cultural Sentiment Analysis
An intelligent NLP system that classifies sentiment toward Kashmiri cultural identity on social media using Machine Learning.

## Overview
The cultural identity of the Kashmiri people expressed through the conventional clothing and food, art, poetry, and language is highly distributed on the social media. But no systematic analysis and interpretation exists on the feedback and feeling of such cultural contents. Users of social media give out opinions that are either good, indifferent or bad, affecting the cultural narrative creation and consumption in the digital era. This topic is much more important for me specifically because as a Kashmiri, I am curious to learn what ML models define as negative and positive with regards to Kashmir.

How can we automatically classify sentiment expressed toward Kashmiri cultural content on social media using machine learning?

## Tech Stack
- Python, Scikit-learn, NLTK
- TF-IDF Vectorisation (5,000 features)
- Gaussian Naive Bayes & Logistic Regression
- Sentiment140 / Kaggle Twitter dataset

## Models
Logistic Regression : Best performer — interpretable, strong on binary text classification
Gaussian Naive Bayes : Probabilistic baseline — efficient on high-dimensional data
Logistic Regression was selected as the final classifier for its stronger performance and interpretability.

## Pipeline
1. Data Ingestion — Twitter/social media dataset + knowledge injection layer
2. Cleaning — Regex stripping of URLs, handles, HTML, emojis
3. Preprocessing — Tokenisation, stop-word removal, lemmatisation
4. Feature Engineering — TF-IDF vectorisation
5. Training — 80/20 train-test split, GridSearch hyperparameter tuning
6. Evaluation — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Results
Logistic Regression:
Negative  →  Precision: 0.74  Recall: 0.80  F1: 0.77
Positive  →  Precision: 0.79  Recall: 0.72  F1: 0.75
Overall Accuracy: 76.05%

## How to Run
pip install -r requirements.txt
jupyter notebook kashmir_senti.ipynb
