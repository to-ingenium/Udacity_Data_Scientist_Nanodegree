# Customer Review Sentiment Analysis Pipeline

A machine learning project that predicts whether customers recommend products based on their reviews. This project implements a complete ML pipeline for binary classification using NLP techniques and ensemble methods.

## Project Overview

This project analyzes customer reviews to predict product recommendations using:
- **Dataset**: 18,442 anonymized clothing reviews with 8 features
- **Target**: Binary classification (1 = Recommended, 0 = Not Recommended)
- **Approach**: Complete ML pipeline with text preprocessing, feature engineering, and model comparison

## Features

The dataset includes:
- **Clothing ID**: Product identifier (categorical)
- **Age**: Customer age (numerical)
- **Title**: Review title (text)
- **Review Text**: Full review content (text)
- **Positive Feedback Count**: Helpful votes from other customers (numerical)
- **Division Name**: Product division (categorical)
- **Department Name**: Product department (categorical)
- **Class Name**: Product class (categorical)



## Key Insights

- **Satisfaction Rate**: 81.62% of customers recommend products
- **Age Distribution**: Most reviewers are 30-60 years old
- **Text Analysis**: Positive and negative reviews share similar keywords but differ in sentiment
- **Product Analysis**: Dresses are most popular but also generate most complaints

## ML Pipeline

### 1. Text Preprocessing
- Text normalization (lowercase)
- Stopword removal
- Tokenization and lemmatization using spaCy
- TF-IDF vectorization (1000 features, bigrams)

### 2. Feature Engineering
- Text length features (character and word counts)
- Sentiment analysis using TextBlob
- Age group categorization
- Feedback ratio normalization

### 3. Model Training
Three models were compared using RandomizedSearchCV:
- **Logistic Regression** (Best): F1=0.9258, Accuracy=94.03%
- **SVM**: F1=0.9223, Accuracy=93.78%
- **Random Forest**: F1=0.9324, Accuracy=93.51%

### 4. Technical Implementaion
- **Pipeline Architecture**: Complete scikit-learn pipeline with custom transformers
- **Cross-validation**: 3-fold CV with F1 scoring
- **Class Imbalance**: Handled using `class_weight='balanced'`
- **Caching**: Preprocessed data cached for faster model iteration


## Requirements

```toml
dependencies = [
    "pandas>=2.3.2",
    "scikit-learn>=1.7.1",
    "spacy>=3.8.7",
    "textblob>=0.19.0",
    "seaborn>=0.13.2",
    "wordcloud>=1.9.4",
    "numpy>=2.3.2",
    "jupyter>=1.1.1"
]
```

## Installation

1. Clone the repository
2. Install dependencies: `uv sync`
3. Download spaCy model: `python -m spacy download en_core_web_sm`

## Usage

### Training
Run the complete analysis in `code/starter.ipynb` to:
- Explore the dataset
- Train and compare models
- Generate cached preprocessing pipeline

### Prediction
Use `code/predictor_ui.ipynb` for making predictions on new reviews with the trained model.

## Model Performance

| Model | Test Accuracy | Test F1 Score | CV F1 Score |
|-------|--------------|---------------|-------------|
| Logistic Regression | 94.03% | 92.58% | 92.58% |
| SVM | 93.78% | 92.23% | 92.23% |
| Random Forest | 93.51% | 93.24% | 93.24% |

The final model achieves excellent performance with balanced precision and recall for both classes.



## Files Generated

- `best_model_pipeline.pkl`: Complete trained pipeline ready for production
- `preprocessing_pipeline.pkl`: Preprocessing steps only
- Processed training/test data cached as pickle files