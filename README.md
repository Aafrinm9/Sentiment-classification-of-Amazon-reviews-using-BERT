# Sentiment Classification of Amazon Reviews Using BERT

This project focuses on classifying the sentiment of Amazon product reviews by leveraging the BERT (Bidirectional Encoder Representations from Transformers) model. The analysis encompasses data extraction, preprocessing, exploratory data analysis (EDA), and sentiment prediction using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Data Source](#data-source)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Sentiment analysis is a crucial aspect of understanding customer opinions and feedback. In this project, we aim to classify Amazon product reviews as positive or negative by utilizing the BERT model, which has set new benchmarks in natural language processing tasks.

## Data Source

The dataset for this project was sourced from the [Amazon Reviews'23](https://amazon-reviews-2023.github.io) repository, curated by the McAuley Lab at UCSD. This comprehensive dataset includes:

- User Reviews: Ratings, text, helpfulness votes, etc.
- Item Metadata: Descriptions, price, raw images, etc.
- Links: User-item and bought-together graphs.

The dataset spans from May 1996 to September 2023, encompassing over 571 million reviews across 33 categories.

## Data Preprocessing

The original data was in JSON format. Key preprocessing steps included:

1. **Transformation**: Converted JSON data into CSV format to facilitate machine learning processes.
2. **Cleaning**: Removed duplicates, handled missing values, and standardized text data.
3. **Tokenization**: Employed BERT's tokenizer to convert textual data into token IDs suitable for model input.
Note: Check JSON 2 CSV.ipynb for the code

## Exploratory Data Analysis

An in-depth exploratory data analysis was conducted to understand the dataset's characteristics. Key findings include:

- **Rating Distribution**: Visualized the distribution of product ratings.
- **Review Length**: Analyzed the length of reviews to determine optimal input sizes for the model.
- **Common Words**: Identified frequently occurring terms in positive and negative reviews.
- Note: Check EDA.ipynb for the code

For detailed visualizations and insights, please refer to the attached EDA file.

## Modeling

To predict the sentiment of user reviews, the following approach was adopted:

1. **Feature Extraction with BERT**: Utilized BERT to convert textual data into numerical representations. BERT captures the context of words in a sentence by considering both preceding and succeeding words, making it highly effective for sentiment analysis tasks. :contentReference[oaicite:0]{index=0}  Note: Check BERT.ipynb for the code
   

3. **Classification with Random Forest**: Implemented a Random Forest classifier to predict the sentiment based on BERT's embeddings. The dataset was split into training, validation, and testing sets in a 70:15:15 ratio. Note: Check RandomForest.ipynb for the code

## Results

The model achieved an accuracy score of 83% on the test set, indicating a high level of performance in classifying the sentiment of Amazon reviews.

## Conclusion

This project demonstrates the effectiveness of combining BERT for feature extraction with a Random Forest classifier for sentiment analysis. The high accuracy underscores the potential of this approach in understanding customer sentiments in large-scale datasets.

## References

- [Amazon Reviews'23 Dataset](https://amazon-reviews-2023.github.io)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
