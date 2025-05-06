
---

# 🧠 Image Captioning with Deep Learning

This repository contains implementations of **Image Captioning** using three different deep learning architectures:

- 🌀 **CNN + LSTM** (Baseline: No Attention)
- 🔭 **CNN + Attention (Bahdanau/Luong)**
- 🧬 **Transformer-based Image Captioning**

Each model takes an image as input and generates a descriptive natural language caption using computer vision and natural language processing techniques.


---

## 📊 Dataset

We use the **MS COCO** dataset which contains:
- 🖼️ 8,000 images
- 📝 5 human-annotated captions per image  
- 📦 Available from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Preprocessing includes:
- Tokenization and padding
- Vocabulary thresholding
- Feature extraction using pretrained CNN (InceptionV3/ResNet50)
- Image-text pair formatting

---

## 🚀 Approaches

### 🌀 1. CNN + LSTM (No Attention)
- **Encoder**: CNN (InceptionV3) extracts image features
- **Decoder**: LSTM generates captions from extracted features

> 📉 BLEU-4: **0.1101**, METEOR: 0.2154

---

### 🔭 2. CNN + Attention (Bahdanau/Luong)
- **Encoder**: CNN for spatial image features
- **Attention Mechanism**: Applies attention over spatial features at each time step
- **Decoder**: LSTM generates word tokens dynamically using context vector

> 📈 BLEU-4: **0.3360**, METEOR: 0.2510

---

### 🧬 3. Transformer (Self-Attention)
- End-to-end transformer architecture
- Self-attention used in both encoder and decoder
- Positional encodings for sequence modeling

> 🚀 BLEU-4: **0.5795**, METEOR: 0.3969

---

## 📈 Evaluation Metrics

- **BLEU** (Bilingual Evaluation Understudy)
- **METEOR**, **ROUGE-L**, **CIDEr**, **SPICE**

| Metric              | LSTM/GRU      | Attention      | Transformer    |
|---------------------|---------------|----------------|----------------|
| BLEU-4              | 0.1101        | 0.3360         | 0.5795         |
| METEOR              | 0.2154        | 0.2510         | 0.3969         |
| CIDEr               | 0.2895        | 1.068          | 1.7444         |
| SPICE               | 0.1541        | 0.1810         | 0.3279         |

---
