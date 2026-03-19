# Transfer Learning for NLP with TensorFlow Hub

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![TensorFlow Hub](https://img.shields.io/badge/TF_Hub-0.16.1-FF6F00?logo=tensorflow)](https://tfhub.dev/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Google_Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![GPU](https://img.shields.io/badge/GPU-Tesla_T4-76B900?logo=nvidia)](https://www.nvidia.com/)

## Overview

This project demonstrates the power of **Transfer Learning** for Natural Language Processing (NLP) using pre-trained text embedding models from [TensorFlow Hub](https://tfhub.dev/). The core task is **binary text classification** — identifying *insincere questions* from the [Quora Insincere Questions Classification dataset](https://www.kaggle.com/c/quora-insincere-questions-classification/data).

Rather than training text embeddings from scratch, pre-trained TF-Hub modules are plugged in as the first layer of a Keras model, enabling strong generalization even on small subsets of data — a key motivation behind transfer learning.

Multiple embedding modules are benchmarked side by side, and one model is further improved via **fine-tuning** (allowing the pre-trained embedding weights to update during training). Training metrics are tracked and visualized using both Matplotlib and **TensorBoard**.

## Project Structure
The notebook is organized into **10 tasks**:

| Task | Description |
|------|-------------|
| Task 1 | Introduction to the Project |
| Task 2 | Setup TensorFlow and Colab Runtime |
| Task 3 | Load the Quora Insincere Questions Dataset |
| Task 4 | TensorFlow Hub for NLP — Concepts & Module Selection |
| Tasks 5 & 6 | Define Function to Build and Compile Models |
| Task 7 | Train Various Text Classification Models |
| Task 8 | Compare Accuracy and Loss Curves |
| Task 9 | Fine-tune Model from TF Hub |
| Task 10 | Train Bigger Models and Visualize Metrics with TensorBoard |

## Dataset

**Quora Insincere Questions Classification**

- **Source:** [Kaggle](https://www.kaggle.com/c/quora-insincere-questions-classification/data) (archived copy loaded directly from [archive.org](https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip))
- **Total records:** 1,306,122 question-label pairs
- **Columns:** `qid`, `question_text`, `target`
- **Label:** Binary — `0` (sincere) / `1` (insincere)
- **Class imbalance:** The dataset is heavily skewed towards sincere questions (`0`), which is clearly reflected in the target distribution histogram generated during EDA.

### Data Split

Stratified sampling is used to preserve the class ratio across splits:

| Split | Size |
|-------|------|
| Training set (1% of full data) | 13,061 rows |
| Validation set (0.1% of remainder) | 1,293 rows |

> Stratified splitting via `sklearn.model_selection.train_test_split` ensures the minority class (insincere questions) is proportionally represented in both training and validation sets despite the heavy imbalance.

## Environment & Dependencies

| Package | Version |
|---------|---------|
| TensorFlow | 2.19.0 |
| TensorFlow Hub | 0.16.1 |
| TensorFlow Datasets | latest |
| TensorFlow Docs | latest (from GitHub) |
| NumPy | latest |
| Pandas | latest |
| Scikit-learn | latest |
| Matplotlib | latest |

**Hardware:** NVIDIA Tesla T4 GPU (Google Colab, Driver: 580.82.07, CUDA 13.0)

## TF-Hub Embedding Modules Used

Five pre-trained text embedding modules from TensorFlow Hub are evaluated:

| Module Name | Embedding Dim | Hub Layer Params | Description |
|-------------|:-------------:|:----------------:|-------------|
| `gnews-swivel-20dim` | 20 | ~400K | Token-based embeddings trained on Google News corpus (Swivel algorithm) |
| `nnlm-en-dim50` | 50 | ~48M | Neural Network Language Model on English Google News corpus |
| `nnlm-en-dim128` | 128 | ~124M | Larger NNLM with higher-dimensional embeddings |
| `universal-sentence-encoder` | 512 | ~256M | Transformer-based sentence encoder; encodes full sentences |
| `universal-sentence-encoder-large` | 512 | ~147M | Larger Transformer-based USE variant |

> These modules serve as the **first layer** of each Keras model, eliminating the need for custom text preprocessing and embedding steps.

## Model Architecture

All models share a common Sequential architecture, with only the TF-Hub embedding layer swapping out per experiment:

```
Input (raw text strings)
        │
        ▼
┌──────────────────────────────┐
│  TF-Hub KerasLayer            │  ← Pre-trained embedding (frozen or fine-tuned)
│  Output shape: [embed_size]   │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Dense(256, activation=relu)  │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Dense(64, activation=relu)   │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Dense(1, activation=sigmoid) │  ← Binary classification output [0, 1]
└──────────────────────────────┘
```

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Metric | Binary Accuracy |
| Epochs | 100 (with early stopping) |
| Batch Size | 32 |
| Callbacks | `EpochDots` (progress display), `EarlyStopping` (overfitting prevention) |

- **`tfdocs.modeling.EpochDots`** — prints a dot per epoch for clean, minimal training progress output.
- **`tf.keras.callbacks.EarlyStopping`** — monitors validation loss and halts training when performance stops improving, preventing overfitting on the small training subset.

## Tasks Walkthrough

### Task 2 — Environment Setup
- GPU availability is verified using `nvidia-smi` (Tesla T4 detected) and `tf.config.list_physical_devices('GPU')`.
- The TensorFlow version (`2.19.0`) and Hub version (`0.16.1`) are printed for reproducibility.
- TensorBoard log directory is initialized using `pathlib` and `tempfile`.

### Task 3 — Dataset Loading & EDA
- The Quora dataset is loaded directly from a remote `.zip` archive into a Pandas DataFrame (1,306,122 rows × 3 columns).
- A **histogram of the `target` column** is plotted, revealing the heavy class imbalance — the vast majority of questions are labeled sincere (`0`).
- The dataset is split into stratified training (13,061 rows) and validation (1,293 rows) subsets using `train_test_split`.
- Sample question texts and labels from the training set are inspected.

### Task 4 — TF Hub for NLP: Concepts
This task covers the conceptual background for two families of text representations:

**Word-based representations:**
- Word2Vec — 300-dim vectors, trained on 300M-token news corpus.
- GloVe — 200-dim vectors, trained on 27B-token tweet corpus.
- Represent questions as the average of individual word embedding vectors.

**Context-based representations:**
- **ELMo** — Character-based, bidirectional LSTM, 1024-dim output.
- **Universal Sentence Encoder (USE)** — Transformer architecture, full sentence encoding, 512-dim output.
- **NNLM** — Simultaneously learns word and sentence representations.

### Tasks 5 & 6 — Model Builder Function
A reusable `train_and_evaluate_model()` function is defined that:
- Accepts `module_url`, `embed_size`, `name`, and `trainable` (default `False`) parameters.
- Builds and compiles the Sequential Keras model described above.
- Prints the model summary.
- Fits the model on the training data with callbacks.
- Returns the training `History` object for later comparison.

### Task 7 — Training Multiple Models
Three models are trained sequentially and their histories stored in a `histories` dictionary:
- `gnews-swivel-20dim` (20-dim, ~400K hub params)
- `nnlm-en-dim50` (50-dim, ~48M hub params)
- `nnlm-en-dim128` (128-dim, ~124M hub params)

### Task 8 — Comparing Accuracy & Loss Curves
`tfdocs.plots.HistoryPlotter` is used to overlay training and validation **accuracy** and **loss** curves for all three models on a single plot per metric, enabling direct visual comparison across architectures.

### Task 9 — Fine-Tuning
The `gnews-swivel-20dim` model is re-trained with `trainable=True`, allowing the embedding weights from TF Hub to be updated during backpropagation. The fine-tuned variant (`gnews-swivel-20dim-finetuned`) is added to the `histories` dict and included in updated comparison plots.

### Task 10 — Bigger Models & TensorBoard
The two Universal Sentence Encoder models are trained:
- `universal-sentence-encoder` — 512-dim Transformer encoder (~256M params)
- `universal-sentence-encoder-large` — 512-dim larger Transformer variant (~147M params)

All six model histories are plotted together. TensorBoard is launched inline using the `%tensorboard` magic command.

## Results & Visualizations

### Target Distribution Histogram
A histogram of the `target` column (Task 3) clearly illustrates the **class imbalance** in the Quora dataset — a strong majority of questions are labeled sincere (`0`), with insincere questions (`1`) forming a small minority. This contextualizes the need for stratified splitting.

### Accuracy Curves (Tasks 8, 9, 10)
Training and validation accuracy are plotted for all models using `tfdocs.plots.HistoryPlotter`. Key observations:
- Curves are updated progressively across tasks as new models are added.
- Context-based models (USE, USE-Large) generally achieve stronger accuracy due to their deeper sentence-level representations.
- Fine-tuning the `gnews-swivel-20dim` model can shift its accuracy trajectory compared to the frozen version.

### Loss Curves (Tasks 8, 9, 10)
Loss curves complement the accuracy plots, showing the convergence behavior of each model. These help identify:
- Models that overfit (training loss drops while validation loss plateaus or rises).
- The effectiveness of early stopping in preventing runaway overfitting on the small training subset.

## Fine-Tuning

Fine-tuning is enabled by setting `trainable=True` on the `hub.KerasLayer`. This allows gradient updates to flow through the pre-trained embedding weights, adapting the embedding space to the vocabulary and writing style of Quora questions.

```python
hub_layer = hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size],
                           dtype=tf.string, trainable=True)  # Fine-tuning enabled
```

The fine-tuned `gnews-swivel-20dim` model's accuracy and loss curves are compared side by side with its frozen counterpart in the multi-model plots.

## TensorBoard

TensorBoard is used in Task 10 to visualize training metrics for all models interactively.

```python
%load_ext tensorboard
%tensorboard --logdir {logdir}
```

The log directory (`logdir`) is initialized at the start of the notebook using `pathlib` and `tempfile`, and cleaned up on each run with `shutil.rmtree`. The inline TensorBoard widget in Colab provides a live dashboard for exploring per-epoch accuracy and loss.

## Learning Objectives
This project demonstrates the ability to:
- Load and use pre-trained NLP text embedding models from TensorFlow Hub as Keras layers.
- Build, compile, and train Sequential text classification models using the `tf.keras` API.
- Compare the performance of multiple embedding strategies — from lightweight word-based models to large Transformer-based sentence encoders.
- Apply fine-tuning to allow pre-trained embedding weights to adapt to a target dataset.
- Visualize and interpret training accuracy/loss curves across multiple models using `tfdocs.plots.HistoryPlotter`.
- Use TensorBoard to monitor and explore training metrics interactively inside a Colab notebook.

---
