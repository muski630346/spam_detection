# spam_detection
#  Spam Detection using Machine Learning

This project builds a text classification model to detect whether an email is **spam** or **ham** (not spam) using natural language processing techniques and machine learning algorithms like **Naive Bayes** and **SVM**.

---

# Problem Statement

Email spam can waste user time and pose phishing risks. Automatically detecting spam helps improve inbox hygiene and user safety.

---

# Objective

Build a model that classifies email messages as **spam** or **ham** using TF-IDF features and train/test split.

---

# Dataset

- **Source**: [Kaggle – Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **File Used**: `spam.csv`
- Two columns:
  - `v1`: Label (ham/spam)
  - `v2`: Message text

---

# Tools & Libraries

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF Vectorizer  
- Naive Bayes & SVM models

---

# Model Pipeline

1. Load and clean dataset  
2. Remove unnecessary columns  
3. Preprocess text:
   - Lowercase, remove stopwords
   - Tokenization
   - TF-IDF vectorization
4. Train/test split  
5. Train models:
   - **Multinomial Naive Bayes**
   - **Support Vector Machine (SVM)**
6. Evaluate using accuracy, precision, recall, confusion matrix

---

# Results

| Model     | Accuracy | Precision | Recall |
|-----------|----------|-----------|--------|
| Naive Bayes | ~98% ✅ | High      | High   |
| SVM         | ~97%    | High      | High   |

---

# How to Run

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn
