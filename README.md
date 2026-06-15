# Social Media Sentiment Engines: Lexicon vs. Machine Learning

A comparative analysis of natural language processing (NLP) approaches for classifying short-form, noisy social media telemetry (Twitter/X).

![Positive Sentiment Word Cloud](images/wordcloud.png)

## Objective & Motivation
In the media intelligence sector, choosing the right sentiment analysis engine is a critical architectural decision. This project evaluates the trade-offs between Traditional Rule-Based Analyzers and a Trained Machine Learning Model.

The goal is to determine which engine provides the most reliable classification for unstructured social media text, evaluating how each handles nuance, slang and neutral-class boundaries.

## Engineering & Methodology
* **Custom Telemetry Normalization:** Twitter data is notoriously noisy. Used a custom regex-based text preprocessor. Augmented standard NLTK stopword lexicons with Twitter-specific noise tokens.
* **EDA:** Conducted exploratory data analysis, including class distributions and lexical frequency mapping.
* **Model Evaluation:** Compared engines across Accuracy, Precision, Recall and F1-Score. Used various visualizations to identify class-specific failure points.

## Key Insights
1. Rule-based analyzers (like VADER) heavily rely on explicit sentiment-bearing words, causing them to frequently misclassify "neutral" text as "positive."
2. The Logistic Regression model (with TF-IDF features) significantly outperformed the rule-based systems, demonstrating that training on domain-specific n-grams is necessary to capture sarcasm, slang and context in short-form social media posts.
3. Error analysis revealed that standard `NLTK` stopword lists aggressively stripped negation modifiers (e.g., "not"). This inverted the semantic meaning of phrases (e.g., "not good" -> "good"), causing unanimous false-positives across all models. This proves that for sentiment analysis, preprocessing must preserve structural context rather than blindly pruning common words.

## Dataset

The dataset is sourced from [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset). 