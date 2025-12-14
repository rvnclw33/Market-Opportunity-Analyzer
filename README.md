# 🎯 Market Opportunity Analyzer

> **Transforming unstructured customer feedback into actionable product strategy using Deep Learning.**

## 📖 Project Overview
The **Market Opportunity Analyzer** is an end-to-end NLP pipeline designed to help Product Managers extract insights from thousands of raw reviews. 

To ensure rigorous Deep Learning Engineering, I implemented a scientific benchmarking strategy, comparing **Unsupervised**, **Classical**, and **Deep Learning** approaches to solve the problem of unstructured data.

## 🧠 Modeling Strategy (The 3 Models)
I architected the solution by benchmarking three distinct modeling approaches:

### 1. Unsupervised Learning (Topic Discovery)
* **Model:** K-Means Clustering ($k=5$)
* **Purpose:** Since the data had no topic labels, I used the **Elbow Method** to find the optimal number of clusters. This automatically buckets reviews into categories like *"UI Issues"* or *"Pricing"* without human supervision.

### 2. Classical Machine Learning (The Baseline)
* **Model:** Logistic Regression
* **Purpose:** Served as a scientific baseline to measure the true performance gain of Deep Learning.
* **Limitation:** It struggled with negation and context (e.g., *“not bad”* was often misclassified).

### 3. Deep Learning (The Core Architecture)
* **Model:** Bi-Directional LSTM (TensorFlow/Keras)
* **Architecture:**
    * **Custom Embeddings:** Trained from scratch (32-dim) to learn domain synonyms.
    * **Bi-Directional Layer:** Reads text forwards and backwards to capture context.
    * **Regularization:** Dropout (0.5) and Early Stopping.

---

## 🚀 Key Features & Innovations
* **Hybrid Ensemble Inference:** The Deep Learning model suffered from "Optimism Bias" due to class imbalance. I engineered a logic layer that detects high-risk keywords (e.g., "cost", "crash") and overrides the Neural Network if necessary.
* **Automated Integrity:** Custom Unit Tests (PyTest) ensure the preprocessing pipeline handles edge cases like emojis and URLs correctly.
* **Knowledge Distillation:** Used **VADER** (lexicon-based) to generate "Silver Labels" for the training data.

## 📊 Benchmarking Results

| Model Approach | Type | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Classical | Fast, Explainable | Fails on sarcasm/negation |
| **Bi-Directional LSTM** | Deep Learning | High Context Awareness | Biased by class imbalance |
| **LSTM + Hybrid Logic** | **Ensemble** | **Best Accuracy & Recall** | Higher inference latency |

## 💻 Tech Stack
* **Core:** Python 3.9+
* **Deep Learning:** TensorFlow, Keras
* **NLP:** NLTK, Scikit-Learn (TF-IDF), VADER
* **App:** Streamlit
* **DevOps:** PyTest
