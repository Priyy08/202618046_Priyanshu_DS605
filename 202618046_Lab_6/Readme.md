* **Name : Madi Priyanshu**
* **ID: 202618046**

---
# Lab Assignment - 6 : Feature Extraction and Machine Learning with Image and Text Data

# PART A :- Asphalt Road Crack Detection System

An end-to-end Computer Vision and Machine Learning pipeline designed to classify asphalt surfaces as **Cracked** or **Non-Cracked** using hand-crafted statistical intensity features and edge metrics.

---

## 1. Project Overview & Architecture

Automated pavement distress assessment enables scalable, cost-efficient infrastructure maintenance. This repository implements two distinct classical Computer Vision workflows:

1. **Baseline Pipeline:** Simple global intensity metrics with median-threshold Canny edge detection.
2. **Optimized Pipeline:** Edge-preserving bilateral filtering, Otsu-guided adaptive hysteresis thresholds, and distribution skewness extraction.

Both pipelines strictly utilize **OpenCV**, **NumPy**, and **scikit-learn** without relying on deep learning architectures or PIL.


## 2. Experimental Setup

To guarantee a fair benchmark (A/B testing), all variables were held strictly identical across both experiments:

* **Total Dataset Size:** 400 images (200 Cracks, 200 NonCracks)
* **Dataset Partition:** Stratified 80/20 split (320 Train images, 80 Test images)
* **Class Balance in Test Split:** 40 Non-Crack images, 40 Crack images
* **Image Input Resolution:** Standardized to 224 x 224 pixels
* **Candidate Classifiers:** Random Forest, Support Vector Classifier (RBF kernel), Logistic Regression

---

## 3. Baseline vs. Optimized Performance Comparison

### Metric Comparison Table

| Metric | Baseline Pipeline | Optimized Pipeline | Delta / Improvement |
| --- | --- | --- | --- |
| **Input Features** | 9 features | 8 features | -1 feature (Noise pruned) |
| **Top Classifier** | Random Forest | Random Forest | Consistent architecture |
| **Validation / Test Accuracy** | 92.50% | **93.75%** | **+1.25%** |
| **Precision** | 0.9048 | **0.9268** | **+2.20%** |
| **Recall** | 0.9500 | **0.9500** | Preserved (High sensitivity) |
| **F1-Score** | 0.9268 | **0.9383** | **+1.15%** |
| **False Positives (Clean classified as Crack)** | 4 | **3** | **-25% reduction in false alarms** |
| **False Negatives (Missed Cracks)** | 2 | **2** | Preserved |
| **Total Test Inference Time (80 samples)** | 0.025953 s | **0.020426 s** | **~21.3% reduction** |
| **Latency per Sample** | 0.3244 ms | **0.2553 ms** | Real-time throughput |

### Confusion Matrix Comparison

```text
Baseline Pipeline Confusion Matrix:
                 Predicted Non-Crack   Predicted Crack
Actual Non-Crack:       36                    4
Actual Crack:            2                   38

Optimized Pipeline Confusion Matrix:
                 Predicted Non-Crack   Predicted Crack
Actual Non-Crack:       37                    3
Actual Crack:            2                   38
```

---

## 4. Engineering Changes in the Optimized Pipeline

The performance gain (+1.25% Accuracy, +2.20% Precision, and a 25% drop in False Positives) was achieved through three justified modifications:

### A. Edge-Preserving Bilateral Smoothing (`cv2.bilateralFilter`)

* **The Problem:** Asphalt surfaces are coarse and comprised of gravel aggregate and bitumen binder. In the baseline pipeline, standard Canny detection captures surface roughness as high-frequency edge noise, causing false positives on clean roads. so just by understanding domain knowledge of problem helps us to apply problem specific techniques to reduce false positive.
* **The Solution:** A bilateral filter ($d=7, \sigma_{\text{Color}}=50, \sigma_{\text{Space}}=50$) was introduced. Unlike standard Gaussian blurring, bilateral filtering smooths radiometric grain within flat asphalt patches while preserving distinct crack line boundaries.

### B. Dynamic Otsu-Guided Canny Hysteresis Thresholding

* **The Problem:** The baseline approach computed Canny thresholds using an empirical offset around the median pixel value (`(1 ± 0.33) * median`). This fixed spread is rigid under varying illumination and fails when shadows dominate.thus it takes account of shadows and spread of cracks as well
* **The Solution:** The optimized pipeline runs Otsu's thresholding (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`) over the filtered image to compute the bimodal luminance separation between the road surface and dark fissures:

$$\text{Upper Threshold} = T_{\text{Otsu}}, \quad \text{Lower Threshold} = 0.5 \times T_{\text{Otsu}}$$

This dynamic boundary eliminates edge noise on clean asphalt and isolates continuous crack paths.

### C. Feature Pruning and Skewness Extraction

* **Pruned Features:** Image `min` and `max` pixel intensities were discarded due to their vulnerability to sensor noise and specular aggregate glints.
* **Added Feature (Skewness):** Pavement fissures act as narrow, localized shadow sinks. We added third-moment distribution skewness ($E[((X - \mu)/\sigma)^3]$) using `scipy.stats.skew`. Cracks create an extended left tail in the pixel histogram, providing a clean statistical separation metric. so addition of new metric cotributes in better crack detection.






---
&nbsp;&nbsp;
# PART B :- Email Spam Classification via Text Vectorization & Pipeline Optimization

An end-to-end Machine Learning project demonstrating email text vectorization with `CountVectorizer`, baseline performance benchmarking, and pipeline optimization strategies to elevate spam detection accuracy to **97.78%** while cutting inference latency.

---


## Project Overview

Spam filtering is a classical high-dimensional natural language processing (NLP) problem. The primary goals of this project are:
- Load and parse pre-extracted frequency features into clean string text.
- Evaluate baseline representations using raw token counts (`binary=False`) versus binary presence (`binary=True`) via `CountVectorizer`.
- Benchmark predictive accuracy, precision, recall, F1-scores, training latency, and prediction (inference) latency.
- Design an optimized pipeline addressing vocabulary noise, multi-word context, vector scaling, and linear margin separation.

---

## Dataset Overview & Class Distribution

The dataset contains **5,172 email records** across 3,002 columns (3,000 word columns, an identifier column `Email No.`, and the target label `Prediction`).

### Class Breakdown
- **Legitimate (Ham / Class 0):** 3,672 emails (71.00%)
- **Spam (Class 1):** 1,500 emails (29.00%)
- **Total Samples:** 5,172

```text
Prediction
0    3672 (71.00%)
1    1500 (29.00%)
```

The 80/20 train-test split (stratified by target class) yields:
- **Training Set:** 4,137 emails
- **Testing Set:** 1,035 emails (735 Ham, 300 Spam)

---


## Phase 1: Baseline Pipeline (Normal Implementation)

In the standard baseline implementation, email representations are generated via `CountVectorizer` and classified using `MultinomialNB`. Two distinct vector representations were compared:

### Term Frequency vs. Binary Occurrence

1. **Term Frequencies (`binary=False`):** Encodes the exact count of each word in the email.
2. **Binary Occurrence (`binary=True`):** Encodes a binary indicator ($1$ if word is present, $0$ otherwise).

### Baseline Results

| Representation | Vocab Size | Accuracy | Spam Recall | Spam Precision | Train Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CountVectorizer (Term Frequencies)** | 2,974 | 94.20% | 0.94 | 0.87 | 0.00735 | 0.00186 |
| **CountVectorizer (Binary Occurrence)** | 2,974 | 93.82% | 0.95 | 0.85 | 0.00616 | 0.00213 |

#### Observations
- Both configurations share 2,974 vocabulary features.
- Frequency counting achieved a slightly higher accuracy (**94.20% vs. 93.82%**) because the repetition of specific corporate or domain-specific words (e.g., *"enron"*, *"deal"*, *"meter"*) helps prevent false positives in legitimate mail.
- Binary occurrence yielded slightly higher spam recall (**0.95 vs. 0.94**), indicating that the presence of high-risk keywords is often informative on its own.

---

## Phase 2: Optimization Strategies

To improve classification accuracy, reduce false positives, and lower prediction latency, several targeted optimizations were integrated:

### 1. Stop-Word Removal (`stop_words='english'`)
- Standard English stopwords (e.g., *"the"*, *"and"*, *"is"*, *"in"*) are evenly distributed across both spam and ham, introducing non-discriminative noise.
- Removing stopwords filters out non-informative dimensions.

### 2. N-Gram Range Expansion (`ngram_range=(1, 2)`)
- Unigram models evaluate words independently, missing critical context.
- Including bigrams enables the pipeline to detect high-signal phrases like `"click here"`, `"free cash"`, or `"credit card"` that strongly indicate spam.

### 3. Vocabulary Capping & Pruning (`min_df=2`, `max_df=0.95`, `max_features=2500`)
- **`min_df=2`:** Prunes single-occurrence typos or anomalous tokens that cause overfitting.
- **`max_df=0.95`:** Eliminates words appearing in almost all documents.
- **`max_features=2500`:** Enforces an upper bound on feature dimensionality, decreasing memory consumption and inference cost.

### 4. L2 Feature Normalization (`Normalizer(norm='l2')`)
- Raw token frequencies scale with email length, which can bias predictions on long documents.
- L2 Normalization projects each sample vector onto a unit sphere ($\|\mathbf{x}\|_2 = 1$), ensuring comparisons reflect word distribution rather than document length.

### 5. Transition to Linear Support Vector Classifier (`LinearSVC`)
- **The Naive Bayes Bottleneck:** Naive Bayes relies on conditional independence:
  $$P(X_1, X_2 \mid Y) = P(X_1 \mid Y) \cdot P(X_2 \mid Y)$$
  Adding bigrams directly violates this assumption because bigrams (`"free cash"`) naturally correlate with their constituent unigrams (`"free"`, `"cash"`). This explains why `Optimized MNB` plateaued at **94.01%**.
- **The LinearSVC Advantage:** Linear SVMs do not assume feature independence. Instead, they identify the maximum-margin hyperplane separating classes in high-dimensional sparse space:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$
  LinearSVC accounts for feature interactions and makes predictions via a single sparse dot product ($\mathbf{w}^T \mathbf{x} + b$), reducing inference latency.

---

## Comparative Benchmarks & Results

### Comprehensive Performance Table

| Pipeline Model | Vocab Size | Accuracy | Macro F1 | Spam F1 | Train Time (ms) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Vanilla CV + MNB)** | 2,974 | 94.20% | 0.9300 | 0.9042 | 7.64 | 1.81 |
| **Optimized MNB (N-Grams + Vocab Limit)** | 2,500 | 94.01% | 0.9300 | 0.9025 | 8.31 | 2.89 |
| **Optimized LinearSVC (N-Grams + L2 + SVM)** | **2,500** | **97.78%** | **0.9700** | **0.9625** | 189.26 | **0.99** |

### Detailed Classification Metrics Comparison

```text
========================= Baseline (Vanilla CV + MNB) =========================
              precision    recall  f1-score   support

     Ham (0)       0.98      0.94      0.96       735
    Spam (1)       0.87      0.94      0.90       300

    accuracy                           0.94      1035

========================= Optimized LinearSVC =========================
              precision    recall  f1-score   support

     Ham (0)       0.99      0.98      0.98       735
    Spam (1)       0.94      0.98      0.96       300

    accuracy                           0.98      1035
```

### Analysis of Gains
1. **Error Reduction:** Overall classification error dropped from **5.80% down to 2.22%**—an error reduction exceeding **61%**.
2. **Spam Precision Improvement:** Spam precision rose from **0.87 to 0.94**, significantly reducing false positives (legitimate messages flagged as spam).
3. **Spam Recall Improvement:** Spam recall climbed to **0.98**, allowing only 2% of junk emails past the filter.
4. **Inference Latency:** Inference time decreased by **45%** (from 1.81 ms to 0.99 ms) due to the reduced vocabulary dimension and fast dot-product evaluation in `LinearSVC`.

---

## Conclusion & Key Takeaways

1. **Feature Engineering vs. Model Synergy:** Here adding bi grams only doesnt improve Naive Bayes performance due to feature correlation. Pairing n-grams with maximum-margin discriminative classifiers (`LinearSVC`) combination enables these richer contextual features to be utilized effectively which is an part of domain aligned feature engineering designed for textual data.
2. **L2 Normalization is Critical for Variable Document Lengths:** Introduces Normalizations of token frequencies ensures that longer emails with naturally higher word counts do not skew distance-based decision boundaries.
3. **Optimized Inference:** Pruning redundant features via `max_features=2500` coupled with a linear decision boundary lowered inference latency below 1 ms, making the pipeline well-suited for production email filtering bringing. here specially use of linearSVC pipeline helped to reduce inference time ranging from 45.3% to 53.5% from its original version.