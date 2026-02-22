# Fashion Forward Forecasting — Pipeline Project

## Project Overview

This project was completed as part of the Udacity Data Science Nanodegree. The goal is to build a machine learning pipeline for **StyleSense**, a rapidly growing online women's clothing retailer, to predict whether a customer would **recommend a product** based on their review and other metadata.

The pipeline handles **numerical**, **categorical**, and **text** data in a single unified workflow — from raw data preprocessing through NLP feature engineering to model prediction and evaluation.

## Dataset

The dataset contains **anonymized women's clothing reviews** with the following features:

| Feature | Type | Description |
|---|---|---|
| Clothing ID | Integer | Identifier for the product being reviewed |
| Age | Integer | Reviewer's age |
| Title | String | Review title |
| Review Text | String | Full review body |
| Positive Feedback Count | Integer | Number of other customers who found the review helpful |
| Division Name | Categorical | High-level product division |
| Department Name | Categorical | Product department |
| Class Name | Categorical | Product class |
| **Recommended IND** | **Binary (Target)** | **1 = Recommended, 0 = Not Recommended** |

## Approach

### Pipeline Architecture

The solution uses a scikit-learn `Pipeline` with a `ColumnTransformer` that processes all data types in parallel:

1. **Numerical features** (Age, Positive Feedback Count) → `SimpleImputer` (median) → `StandardScaler`
2. **Categorical features** (Division Name, Department Name, Class Name) → `SimpleImputer` (most frequent) → `OneHotEncoder`
3. **Text features** (Title + Review Text) → Custom `TextPreprocessor` (spaCy lemmatization, stop word/punctuation removal) → `TfidfVectorizer` (up to 5000 features, unigrams + bigrams)

All preprocessing feeds into a **Logistic Regression** classifier, and the entire pipeline is hyperparameter-tuned using **GridSearchCV** with 3-fold cross-validation.

### NLP Techniques

- **spaCy** for tokenization, lemmatization, and stop word removal (using `en_core_web_sm`)
- **TF-IDF** vectorization with unigram + bigram features
- **POS tagging** to extract counts of adjectives, nouns, verbs, and adverbs as additional features
- **Named Entity Recognition (NER)** to count brand/product/organization mentions in reviews

### Visualizations

- Target class distribution
- Confusion matrix heatmap
- Top TF-IDF feature importance chart
- POS tag distribution across recommended vs. not recommended reviews

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.8976 |
| Precision | 0.9235 |
| Recall | 0.9545 |
| F1 Score | 0.9388 |

Best hyperparameters found via GridSearchCV: `C=10.0`, `penalty='l2'`, `max_features=5000`

## File Structure

```
dsnd-pipelines-project/
├── README.md                  # This file
├── starter.ipynb              # Main Jupyter Notebook with all code and analysis
├── requirements.txt           # Python dependencies
└── data/
    └── reviews.csv            # Dataset (women's clothing reviews)
```

## How to Run

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/udacity/dsnd-pipelines-project.git
cd dsnd-pipelines-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the Notebook

```bash
jupyter notebook starter.ipynb
```

Run all cells sequentially. The full pipeline (training + GridSearchCV + evaluation) takes approximately 15–20 minutes depending on hardware.

## Libraries Used

- **pandas** & **numpy** — Data manipulation
- **scikit-learn** — Pipeline, preprocessing, model training, evaluation, and hyperparameter tuning
- **spaCy** — NLP text preprocessing (tokenization, lemmatization, POS tagging, NER)
- **matplotlib** & **seaborn** — Visualizations

## Acknowledgments

- Dataset provided by Udacity as part of the Data Science Nanodegree
- Project scenario: StyleSense online women's clothing retailer