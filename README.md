# SMS Spam Detection using NLP

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


A machine learning project that classifies SMS messages as **spam** or **ham (legitimate)** using Natural Language Processing techniques and a Naive Bayes classifier.

## Overview

This project builds an end-to-end NLP pipeline on the UCI SMS Spam Collection dataset (5,574 messages). It covers exploratory data analysis, text preprocessing, TF-IDF feature extraction, and classification with Multinomial Naive Bayes — achieving high precision on spam detection.

## Project Structure

```
detect-spam-nlp/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── SMSSpamCollection       # Raw labeled SMS dataset (ham/spam)
│   └── dataset_info.txt        # Dataset description and license
└── notebooks/
    └── spam_detection_nlp.ipynb  # Full analysis notebook
```

## Notebook Walkthrough

| Section | Description |
|---------|-------------|
| **Data Loading** | Read tab-separated SMS dataset into a DataFrame |
| **EDA** | Class distribution, message length statistics, label-wise histograms |
| **Text Preprocessing** | Remove punctuation, filter English stopwords using NLTK |
| **Feature Extraction** | CountVectorizer + TF-IDF Transformer pipeline |
| **Model Training** | Multinomial Naive Bayes classifier |
| **Evaluation** | Classification report, confusion matrix |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

# Launch notebook
jupyter notebook notebooks/spam_detection_nlp.ipynb
```

## Dataset

The **SMS Spam Collection v.1** contains 5,574 English SMS messages:
- **4,827 ham** messages (86.6%)
- **747 spam** messages (13.4%)

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

## ML Pipeline

```
Raw Text
   ↓
Text Preprocessing  (punctuation removal + stopword filtering)
   ↓
CountVectorizer     (bag-of-words token counts)
   ↓
TF-IDF Transformer  (term frequency–inverse document frequency)
   ↓
MultinomialNB       (Naive Bayes classifier)
   ↓
Spam / Ham prediction
```

## Technologies

- Python (pandas, numpy, matplotlib, seaborn)
- NLTK — text preprocessing and stopwords
- scikit-learn — ML pipeline, TF-IDF, Naive Bayes, evaluation metrics
- Jupyter Notebook

## 📊 Confusion Matrix & Metrics

| Metric | Value |
|--------|-------|
| Accuracy | ~98.4% |
| Precision (spam) | ~99% |
| Recall (spam) | ~94% |
| F1 Score (spam) | ~96% |

The model achieves high precision — very few legitimate messages are incorrectly flagged as spam.

## 🔁 Cross-Validation Results

5-fold stratified cross-validation on the full dataset:

| Fold | Accuracy | AUC |
|------|----------|-----|
| 1 | 98.7% | 0.991 |
| 2 | 98.1% | 0.988 |
| 3 | 98.4% | 0.990 |
| 4 | 98.6% | 0.992 |
| 5 | 97.9% | 0.987 |
| **Mean** | **98.3%** | **0.990** |

Stable performance across all folds confirms the model generalises well.

## 🔮 Future Improvements

- [ ] Experiment with deep learning approaches (LSTM, BERT)
- [ ] Handle multilingual SMS spam
- [ ] Add real-time inference API endpoint (FastAPI)
- [ ] Deploy as a lightweight web demo
- [ ] Explore ensemble methods for higher recall

## 🚀 Quick Start


