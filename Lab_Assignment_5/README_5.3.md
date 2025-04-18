# 🐦 Twitter Sentiment Analysis using LSTM & BiLSTM

A deep learning project that classifies tweet sentiments (positive or negative) using LSTM and BiLSTM models implemented in TensorFlow/Keras.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [References](#references)

---

## 🧠 About the Project

This project aims to classify the sentiment of tweets into **positive** or **negative** using deep learning models. We use **LSTM** and **Bidirectional LSTM (BiLSTM)** architectures to capture the contextual meaning and dependencies in tweet sequences.

---

## 📂 Dataset

- Dataset: [Kaggle - Disaster Tweets Sentiment Dataset](https://www.kaggle.com/datasets)
- Features:
  - `id`: Unique identifier
  - `text`: Tweet content
  - `target`: Sentiment label (0 = Negative, 1 = Positive)

---

## 🔧 Preprocessing

- Lowercasing
- Removing:
  - URLs
  - Mentions (@user)
  - Hashtags (#topic)
  - Punctuation & Emojis
- Tokenization
- Padding sequences to uniform length
- Train-Test Split: 80% train, 20% test

---

## 🧱 Model Architecture

### ✅ LSTM Model

- Embedding Layer
- LSTM Layer (128 units)
- Dropout (0.5)
- Dense Output Layer with Sigmoid Activation

### ✅ BiLSTM Model

- Embedding Layer
- Bidirectional LSTM (128 units)
- Dropout (0.5)
- Dense Output Layer with Sigmoid Activation

**Loss Function**: Binary Crossentropy  
**Optimizer**: Adam  
**Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 📈 Performance

| Model     | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| LSTM      | 82.1%    | 81.2%     | 80.6%  | 80.9%    |
| BiLSTM    | 83.6%    | 82.7%     | 82.2%  | 82.4%    |

📌 **BiLSTM outperforms LSTM** in capturing both forward and backward context of the text.

---

