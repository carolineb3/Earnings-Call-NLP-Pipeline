# Earnings-Call-NLP-Pipeline

## Earnings Call Intelligence: Spark Infrastructure and Machine Learning Models

This project builds a big data natural language processing pipeline to test whether language patterns in S&P 500 earnings call transcripts can help predict short-term post-earnings stock movement. The pipeline uses PySpark to transform unstructured financial text into structured sentiment, topic, divergence, and market-based features for machine learning classification.

The core research question is: **Can earnings call language help predict whether a stock moves up or down after the call?**

## Project Overview

The project analyzes FY2021–FY2024 S&P 500 earnings call transcripts and links each call to a 3-day market-adjusted stock return. Market reaction is measured as the company’s 3-day post-earnings return minus the SPY benchmark return.

After removing low-signal neutral market reactions within a ±0.5% market-adjusted return band, the final labeled dataset contains **7,076 earnings call observations** with a nearly balanced positive (50.6%) / negative (49.4%) class split.

## Technologies & Architecture

- **Big Data Framework:** PySpark, Spark DataFrames, Spark MLlib
- **NLP & Deep Learning:** HuggingFace Transformers, `ProsusAI/finbert`, NLTK, Latent Dirichlet Allocation
- **Financial Data:** HuggingFace Datasets, yfinance, SPY benchmark returns
- **Feature Engineering:** Loughran-McDonald dictionary sentiment, FinBERT sentiment, topic shift, lag/trend features, sector controls
- **Explainability:** SHAP and permutation importance
- **Storage:** Parquet checkpointing for reproducible stepwise execution

## Data Sources

The project uses two primary data sources:

1. **Earnings call transcripts**  
   Transcript data is loaded from the HuggingFace dataset `kurry/sp500_earnings_transcripts`. The pipeline filters available transcripts to FY2021–FY2024 and extracts ticker, company name, fiscal quarter, earnings date, transcript text, speaker turns, and sector metadata.

2. **Market reaction data**  
   Stock prices are retrieved through the yfinance API. For each earnings date, the pipeline calculates the company’s 3-day return and subtracts the SPY 3-day return to create a market-adjusted return.

The target variable is created as follows:

- **Positive:** market-adjusted 3-day return greater than +0.5%
- **Negative:** market-adjusted 3-day return less than -0.5%
- **Neutral:** market-adjusted 3-day return between -0.5% and +0.5%, removed from the binary modeling dataset

## The 9-Step Pipeline

The codebase is organized into nine sequential Google Colab notebooks.

### 1. Data Ingestion

Loads FY2021–FY2024 S&P 500 earnings transcripts from HuggingFace and retrieves post-earnings stock price data from yfinance.

Key outputs:

- `Transcripts_JSON/`
- `stock_prices.csv`
- `sector_map.csv`

### 2. Spark Infrastructure

Loads transcript JSON files and stock price data into Spark, joins them by ticker and earnings date, and creates the binary return label.

This step also removes neutral reactions within the ±0.5% market-adjusted return band.

Key output:

- `584_earnings_final.parquet`

### 3. NLP Preprocessing

Prepares transcript text for topic modeling using NLTK.

Processing steps include:

- Tokenization
- Lowercasing
- Punctuation removal
- English stop-word removal
- Financial stop-word filtering
- Single-pass lemmatization

This cleaned text is used for LDA topic modeling, not for Loughran-McDonald dictionary scoring.

Key output:

- `584_earnings_nlp.parquet`

### 4. Loughran-McDonald Dictionary Sentiment

Scores raw prepared remarks and Q&A sections using the Loughran-McDonald financial sentiment dictionary.

The model calculates section-level scores for:

- Positive language
- Negative language
- Uncertainty language
- Litigious language
- Net sentiment

Dictionary scoring is applied to raw transcript text rather than lemmatized text so that financial dictionary word matches are preserved.

Key output:

- `584_earnings_sentiment.parquet`

### 5. FinBERT Deep Learning Sentiment

Uses `ProsusAI/finbert`, a BERT-based financial sentiment model from HuggingFace, to score earnings call sections as positive, negative, or neutral.

Because FinBERT has a 512-token input limit, each prepared remarks section and Q&A section is split into token-length chunks of 450 tokens or fewer. Each chunk is scored separately, and the chunk-level probabilities are recombined using a token-count-weighted average.

This produces section-level FinBERT sentiment features for both prepared remarks and Q&A.

Key output:

- `584_earnings_finbert.parquet`

### 6. LDA Topic Modeling

Uses Latent Dirichlet Allocation to extract recurring topic patterns from prepared remarks and Q&A sections.

This step:

- Fits CountVectorizer vocabulary on training years only
- Fits separate LDA models for prepared remarks and Q&A sections
- Extracts five latent topics
- Computes Jensen-Shannon divergence between consecutive quarters to measure topic shift

Key output:

- `584_earnings_topics.parquet`

### 7. Feature Engineering

Builds the final machine-learning feature matrix.

Feature groups include:

- Loughran-McDonald sentiment scores
- FinBERT sentiment scores
- Prepared remarks vs Q&A divergence features
- LDA topic probabilities
- Q&A topic shift
- Lagged sentiment features
- Sentiment trend features
- Quarter/seasonality controls
- Sector metadata

MinMaxScaler is fit on training years only (2021-2023) to prevent temporal leakage. Sector is retained as a raw categorical field and later one-hot encoded in the machine learning steps.

Key output:

- `584_earnings_features.parquet`

### 8. Machine Learning Models

Trains baseline machine learning models using PySpark MLlib.

Models include:

- Logistic Regression
- Random Forest
- Gradient Boosted Trees
- Majority-vote ensemble

The train/test split is temporal rather than random:

- **Training years:** 2021–2023
- **Test year:** 2024

This avoids look-ahead bias and better reflects the real-world forecasting setting.

Key output:

- `584_earnings_ml_results.parquet`

### 9. Tuning and Explainability

Performs advanced tuning and interpretation for the Gradient Boosted Trees model.

This step includes:

- Expanded GBT hyperparameter tuning
- SHAP explainability using a scikit-learn mirrored model
- Permutation importance
- 30-day pre-earnings momentum and volatility proxy features
- Comparison of language-only features versus language plus price-proxy features

## Leakage Controls

The project uses several safeguards to reduce data leakage:

- Temporal train/test split instead of random splitting
- CountVectorizer fit on training years only
- LDA models fit on training years only
- MinMaxScaler fit on training years only
- Sector one-hot encoding fit on training years only
- 2024 held out as the final test year

## Results

The advanced tuned Gradient Boosted Trees model achieved the following performance on the 2024 temporal test set:

- **Accuracy:** 0.4983
- **Precision:** 0.4980
- **Recall:** 0.4983
- **F1 Score:** 0.4980
- **AUC-ROC:** 0.4835

The best tuned GBT parameters were:

- **maxDepth:** 3
- **maxIter:** 50
- **stepSize:** 0.05

The model performed close to a random classifier. This suggests that short-term post-earnings stock movement is difficult to predict from transcript language alone.

## Interpretation

The project does not support the claim that earnings call language alone can reliably predict short-term stock direction. Instead, the results suggest that public earnings call language may already be quickly incorporated into market prices, while short-term reactions are likely influenced by factors not fully captured in transcript text, such as earnings surprises, forward guidance, analyst expectations, macroeconomic conditions, and firm-specific news.

However, the project successfully demonstrates a scalable framework for processing financial text at scale. The pipeline converts thousands of long earnings call transcripts into structured, model-ready features using Spark, financial dictionary sentiment, transformer-based sentiment, topic modeling, feature engineering, and explainable machine learning.

## Business Use Case

This project is best interpreted as a prototype risk-screening tool rather than a standalone trading model.

The pipeline could help analysts identify earnings calls with:

- Elevated negative language
- Increased uncertainty
- Greater prepared remarks vs Q&A divergence
- Major topic shifts across quarters
- Unusual language-based risk patterns

These signals could be used to prioritize which calls deserve closer human review.

## Repository Structure

```text
Notebook 1 - Data Ingestion
Notebook 2 - Spark Infrastructure
Notebook 3 - NLP Preprocessing
Notebook 4 - Loughran-McDonald Dictionary Sentiment
Notebook 5 - FinBERT Sentiment
Notebook 6 - LDA Topic Modeling
Notebook 7 - Feature Engineering
Notebook 8 - Machine Learning Models
Notebook 9 - Tuning and SHAP
```

## How to Run the Code

Large generated files, including transcript JSONs and Parquet checkpoints, are not included in this repository due to GitHub file size limits.

To recreate the pipeline:

1. Clone this repository or open the notebooks directly in Google Colab.
2. Upload `LoughranMcDonald_MasterDictionary_1993-2024.csv` into the project working directory.
3. Run Notebook 1 to download transcripts and stock price data.
4. Run Notebooks 2 through 9 sequentially.
5. Each notebook writes intermediate outputs as Parquet checkpoints for efficient reruns.

A GPU runtime is recommended for the FinBERT step because full-section transformer scoring requires processing many transcript chunks.

## Authors

Caroline Brady  
Alyssa Prichard  

MIS 584: Big Data Technologies  
University of Arizona