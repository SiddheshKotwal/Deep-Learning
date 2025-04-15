# 🧠 NLP Preprocessing & Text Classification

---

## 🎯 Objective
Implement NLP preprocessing techniques and build a text classification model using machine learning on the SMS Spam Collection dataset.

---

## ✅ Learning Outcomes
- Understand and apply NLP preprocessing (tokenization, stopword removal, stemming, lemmatization)
- Use text vectorization techniques (TF-IDF, CountVectorizer)
- Build and evaluate a machine learning classification model
- Measure performance using standard evaluation metrics

---

## 🗂️ Dataset
- **Name**: SMS Spam Collection
- **Source**: [Best Datasets for Text Classification](https://en.innovatiana.com/post/best-datasets-for-text-classification)

---

## 🛠️ Steps & Tools

### 🔹 Step 1: Import Libraries
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk`, `spaCy`
- `scikit-learn`

### 🔹 Step 2: Load Dataset
- Read TSV file into DataFrame
- Inspect data and label distribution

### 🔹 Step 3: NLP Preprocessing
- Lowercasing
- Remove non-alphabetic characters
- Tokenization (`spaCy`)
- Stopword removal (`nltk`)
- Stemming (`PorterStemmer`)
- Lemmatization (`spaCy`)

### 🔹 Step 4: Vectorization
- `CountVectorizer`
- `TfidfVectorizer`

### 🔹 Step 5: Model Building
- **Algorithm**: Logistic Regression
- **Split**: 80% train / 20% test

### 🔹 Step 6: Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Classification Report

---

## 📊 Results

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 0.9764    |
| Precision  | 0.9591    |
| Recall     | 0.9487    |
| F1 Score   | 0.9539    |

---

## 💡 Conclusion
- Logistic Regression showed high performance for spam detection.
- Proper preprocessing and vectorization significantly improved results.
- Minimal misclassification observed.

---

## 🔮 Future Work
- Explore SVM, Naïve Bayes, and deep learning models (e.g., BERT)
- Apply to multi-class datasets
- Deploy in real-time systems

---