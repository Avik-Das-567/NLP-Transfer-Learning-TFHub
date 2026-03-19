# Transfer Learning for NLP with TensorFlow Hub

A practical deep learning project demonstrating how to leverage pre-trained NLP models from **TensorFlow Hub** for binary text classification using the **Quora Insincere Questions** dataset. The project walks through loading pre-trained text embeddings, building and training classification models, fine-tuning hub modules, and visualizing training metrics with **TensorBoard**, all within a GPU-accelerated Google Colab environment.

## Overview

Transfer learning makes it possible to save training resources and achieve strong model generalization even when training on a relatively small subset of data. This project demonstrates this by training several different TF-Hub embedding modules on a large-scale question classification task, comparing their performance, and fine-tuning the hub layers for improved accuracy.

## Project Structure

The project is organized into **10 tasks** inside a single Jupyter/Colab notebook:

| Task | Description |
|------|-------------|
| **Task 1** | Introduction to the Project |
| **Task 2** | Setup TensorFlow and Colab Runtime (GPU check) |
| **Task 3** | Download and Import the Quora Insincere Questions Dataset |
| **Task 4** | TensorFlow Hub for Natural Language Processing |
| **Task 5 & 6** | Define Function to Build and Compile Models |
| **Task 7** | Train Various Text Classification Models |
| **Task 8** | Compare Accuracy and Loss Curves |
| **Task 9** | Fine-tune Model from TF Hub |
| **Task 10** | Train Bigger Models and Visualize Metrics with TensorBoard |

## Dataset

**Quora Insincere Questions Classification**

- **Source**: [Kaggle](https://www.kaggle.com/c/quora-insincere-questions-classification/data) (via [Internet Archive mirror](https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip))
- **Size**: 1,306,122 rows × 3 columns
- **Task**: Binary classification — predict whether a question is sincere (`0`) or insincere (`1`)
- **Key columns**:
  - `question_text` — the raw question string
  - `target` — binary label (0 = sincere, 1 = insincere)

The dataset is loaded directly into a Pandas DataFrame, split into training and validation sets, and fed into the models as raw text strings (no manual tokenization required — the TF-Hub layers handle this internally).

## Models & Embeddings

Five pre-trained TF-Hub text embedding modules are used and compared:

| Module | Embedding Dim | Type | Notes |
|--------|--------------|------|-------|
| `gnews-swivel-20dim` | 20 | Word-based | Trained on Google News corpus |
| `nnlm-en-dim50` | 50 | Word-based (NNLM) | English Google News 200B corpus |
| `nnlm-en-dim128` | 128 | Word-based (NNLM) | English Google News 200B corpus |
| `universal-sentence-encoder` (v4) | 512 | Context-based (Transformer) | Encodes full sentence semantics |
| `universal-sentence-encoder-large` (v5) | 512 | Context-based (Transformer) | Larger USE variant |

A **fine-tuned** variant of `gnews-swivel-20dim` is also trained with `trainable=True`, allowing the hub layer weights to be updated during backpropagation.

## Model Architecture

All models share the same downstream classifier head, with only the pre-trained hub layer varying:

```
Input (raw text string)
    │
    ▼
Hub Embedding Layer (TF-Hub module, optionally trainable)
    │
    ▼
Dense(256, activation='relu')
    │
    ▼
Dense(64, activation='relu')
    │
    ▼
Dense(1, activation='sigmoid')   ← Binary output
```

**Compilation settings:**

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss      = tf.losses.BinaryCrossentropy()
metrics   = [tf.metrics.BinaryAccuracy(name='accuracy')]
```

**Training callbacks:**

- `EarlyStopping` — monitors `val_loss` with patience of 2 epochs
- `TensorBoard` — logs training metrics per model for comparison
- `EpochDots` — lightweight epoch progress display (via `tensorflow_docs`)

## Results

Training histories for all models are collected and compared side-by-side using `tfdocs.plots.HistoryPlotter`:

- **Accuracy curves** — plotted across epochs for all models
- **Loss curves** — plotted across epochs for all models
- **TensorBoard** — used to visualize and compare metrics for larger models (USE variants) in an interactive dashboard

Fine-tuning the `gnews-swivel-20dim` hub layer (`trainable=True`) makes **all 421,909 parameters trainable** (vs. only 21,889 trainable params in the frozen variant), allowing the embedding weights to adapt to the specific task.

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.19.0 | Core deep learning framework |
| `tensorflow-hub` | 0.16.1 | Pre-trained model repository |
| `tensorflow-datasets` | — | Dataset utilities |
| `tensorflow-docs` | latest | Training plots and epoch callbacks |
| `pandas` | — | Data loading and manipulation |
| `numpy` | — | Numerical operations |
| `matplotlib` | — | Plotting accuracy/loss curves |
| `TensorBoard` | (built-in) | Interactive training visualization |

**Hardware**: NVIDIA Tesla T4 GPU (CUDA 13.0) via Google Colab

## Key Concepts

### Word-based Representations
Combine word embeddings of individual content words in a sentence (e.g., averaging). Examples used in this project:
- **Swivel (gnews-swivel-20dim)** — co-occurrence matrix factorization on Google News
- **NNLM (nnlm-en-dim50/128)** — Neural Network Language Model trained on Google News 200B corpus

### Context-based Representations
Generate a single vector for the entire sentence, taking into account word order and co-occurrence. Examples used:
- **Universal Sentence Encoder (USE)** — Transformer-based architecture; produces 512-dimensional sentence vectors

### Transfer Learning & Fine-Tuning
- **Frozen hub layers** (`trainable=False`): Only the downstream Dense layers are trained. Fast and memory efficient.
- **Fine-tuned hub layers** (`trainable=True`): All weights including the embedding module are updated. Allows task-specific adaptation at the cost of more compute.

## Learning Objectives
By completing this project, we will be able to:
- Use various pre-trained NLP text embedding models from **TensorFlow Hub**
- Perform **transfer learning** to fine-tune models on our own text data
- Build and compile Keras models with **hub layers** as feature extractors
- Apply **early stopping** and **TensorBoard** callbacks during training
- Visualize and compare **accuracy and loss curves** across multiple models
- Understand the trade-offs between word-based and context-based text representations

---
