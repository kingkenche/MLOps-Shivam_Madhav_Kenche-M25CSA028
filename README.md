# Assignment 3: End-to-End HuggingFace Model Training & Docker Deployment

## Project Overview

This project fine-tunes **DistilBERT** (`distilbert-base-cased`) for **book genre classification** on Goodreads reviews from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html). The model classifies reviews into 8 genres:

- Poetry
- Children
- Comics & Graphic
- Fantasy & Paranormal
- History & Biography
- Mystery, Thriller & Crime
- Romance
- Young Adult

## Model Selection Rationale

**DistilBERT (distilbert-base-cased)** was chosen for the following reasons:

1. **Efficiency**: DistilBERT is 60% faster and 40% smaller than BERT-base while retaining 97% of its language understanding capability
2. **Cased variant**: Preserves capitalization, which can be informative for distinguishing genres (e.g., proper nouns in history/biography)
3. **HuggingFace ecosystem**: Seamless integration with the Trainer API, tokenizers, and Hub
4. **Resource-friendly**: Ideal for environments with limited GPU resources (e.g., Google Colab free tier)

## Project Structure

```
Assignment-3/
├── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb  # Original notebook
├── src/
│   ├── __init__.py
│   ├── data.py          # Data loading, sampling, train/test split
│   ├── utils.py         # Dataset class, label encoding, metrics
│   ├── train.py         # Model loading, TrainingArguments, Trainer
│   ├── eval.py          # Evaluation, classification report, confusion matrix
│   └── push_to_hub.py   # Push model to HuggingFace Hub
├── main.py              # Full pipeline: data → train → eval → save
├── eval_only.py         # Eval-only entry point (for production Docker)
├── evaluate_from_hub.py # Load from HF Hub + re-evaluate + compare
├── requirements.txt
├── Dockerfile           # Development Docker image
├── Dockerfile.eval      # Production eval-only Docker image
├── README.md
└── .gitignore
```

## Training Summary

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-cased` |
| Epochs | 3 |
| Train batch size | 10 |
| Eval batch size | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max token length | 512 |
| Train samples | 6,400 (800 per genre × 8 genres) |
| Test samples | 1,600 (200 per genre × 8 genres) |

## Evaluation Results

### Local Model Evaluation (Colab Run)

| Metric | Score |
|--------|-------|
| Accuracy | **60.63%** |
| Precision (weighted) | 61.16% |
| Recall (weighted) | 60.63% |
| F1 (weighted) | 60.70% |
| Loss | 1.259 |

> *Note*: Results vary due to random sampling of Reviews from the sourceURL. The Colab run used a different random subset than the local verification.

### Docker Container Evaluation

| Metric | Score |
|--------|-------|
| Accuracy | **60.75%** |
| Precision (weighted) | 61.29% |
| Recall (weighted) | 60.75% |
| F1 (weighted) | 60.83% |
| Loss | 1.259 |

> Results are saved in `results/eval_results_docker.json` inside the container.

## HuggingFace Hub Model Evaluation (Local Verification)

| Metric | Score |
|--------|-------|
| Accuracy | **70.88%** |
| Precision (weighted) | 70.88% |
| Recall (weighted) | 70.88% |
| F1 (weighted) | 70.78% |
| Loss | 0.918 |

> *Note*: Results vary due to random sampling of the dataset. The Docker run (60.75%) aligns closely with the Colab training (60.6%).

## HuggingFace Model Link

🤗 **Model**: [kingkenche/distilbert-goodreads-genre-classifier](https://huggingface.co/kingkenche/distilbert-goodreads-genre-classifier)

## GitHub Repository

📂 **Repository**: [MLOps-Shivam_Madhav_Kenche-M25CSA028 (Branch: assignment-3)](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028/tree/assignment-3)

To explore the code:
```bash
git clone https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028.git
cd MLOps-Shivam_Madhav_Kenche-M25CSA028
git checkout assignment-3
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Training Pipeline (on Google Colab with GPU)

```bash
python main.py
```

### 3. Push Model to HuggingFace Hub

```bash
python -m src.push_to_hub --hf-token YOUR_HF_TOKEN
```

### 4. Re-evaluate from HuggingFace Hub

```bash
python evaluate_from_hub.py
```

## Docker Instructions

### Development Docker Image (Task 2)

```bash
# Build
docker build -t assignment3 .

# Run (full pipeline — requires GPU for training)
docker run assignment3

# Verify Python and libraries
docker run assignment3 python -c "import transformers; import torch; print('transformers:', transformers.__version__); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Production Eval-Only Docker Image (Task 9)

```bash
# Build
docker build -f Dockerfile.eval -t assignment3-eval .

# Run (pulls model from HuggingFace and evaluates)
docker run assignment3-eval

# Override model repo
docker run -e HF_REPO_ID="kingkenche/distilbert-goodreads-genre-classifier" assignment3-eval
```

## Challenges

1. **No local GPU**: Training was performed on Google Colab (free tier with T4 GPU). Docker is used only for evaluation.
2. **Large data downloads**: Goodreads review files are streamed from UCSD servers and can be slow. A pickle cache is used to avoid re-downloading.
3. **Memory constraints**: Only 1,000 reviews per genre (out of 10,000 loaded) are used for training/testing to fit within Colab memory limits.
4. **Tokenization overhead**: DistilBERT tokenization of 8,000 reviews with max_length=512 requires significant RAM; `DistilBertTokenizerFast` helps mitigate this.

## GitHub Repository

📂 **Repository**: [GitHub Link — to be added after push]

## Author

- **HuggingFace Profile**: [kingkenche](https://huggingface.co/kingkenche)
