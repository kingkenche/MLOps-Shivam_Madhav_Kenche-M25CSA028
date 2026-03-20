# ── Auto-install missing dependencies ────────────────────────────────────────
import subprocess, sys
for pkg in ["optuna", "ray[tune]"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                           "--user", "-q"])
# ─────────────────────────────────────────────────────────────────────────────

# # Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna
# **English → Hindi** | Custom PyTorch Transformer + Ray Tune + Optuna + ASHA
# 
# ---
# | Section | Description |
# |---------|-------------|
# | **Part 0** | Imports, Data, Vocabulary, Architecture (unchanged from baseline) |
# | **Part 1** | Baseline Training — 100 epochs, hardcoded HP |
# | **Part 2** | `train_tune()` refactor + Search Space + ASHA + Optuna Sweep |
# | **Part 3** | Retrain best config, compute final BLEU, save model |

# ── 0-A: Imports ─────────────────────────────────────────────────────────────
import os, time, math, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
nltk.download("punkt", quiet=True)

# Ray Tune
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── 0-B: Load Dataset ────────────────────────────────────────────────────────
# Expects "English-Hindi.tsv" in the working directory (from the Drive link)
df = pd.read_csv("English-Hindi.tsv", sep="\t", header=None, names=["id1", "en", "id2", "hi"])
df = df[["en", "hi"]].dropna().reset_index(drop=True)
print(f"Total pairs: {len(df)}")
df.head()


# ── 0-C: EDA ─────────────────────────────────────────────────────────────────
df["en_len"] = df["en"].apply(lambda x: len(x.split()))
df["hi_len"] = df["hi"].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df["en_len"], bins=30, kde=True, color="skyblue")
plt.title("English Sentence Lengths")
plt.subplot(1, 2, 2)
sns.histplot(df["hi_len"], bins=30, kde=True, color="salmon")
plt.title("Hindi Sentence Lengths")
plt.tight_layout(); plt.show()

print("English:", df["en_len"].describe().to_dict())
print("Hindi  :", df["hi_len"].describe().to_dict())


# ── 0-D: Vocabulary (exact copy from en_to_hi.ipynb) ────────────────────────
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx  = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def tokenize(self, sentence):
        return sentence.lower().strip().split()

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    def __len__(self):   return len(self.stoi)
    def __getitem__(self, token): return self.stoi.get(token, self.stoi["<unk>"])

en_vocab = Vocabulary(freq_threshold=2)
hi_vocab = Vocabulary(freq_threshold=2)
en_vocab.build_vocab(df["en"].tolist())
hi_vocab.build_vocab(df["hi"].tolist())

print(f"English vocab size : {len(en_vocab)}")
print(f"Hindi   vocab size : {len(hi_vocab)}")

SRC_PAD_IDX = en_vocab["<pad>"]
TGT_PAD_IDX = hi_vocab["<pad>"]


# ── 0-E: Encode / Decode helpers ────────────────────────────────────────────
def encode_sentence(sentence, vocab, max_len=50):
    tokens = ([vocab.stoi["<sos>"]]
              + vocab.numericalize(sentence)[:max_len - 2]
              + [vocab.stoi["<eos>"]])
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))


# ## Architecture (exact copy from `en_to_hi.ipynb`)

# ── 0-F-1: Positional Encoding ───────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── 0-F-2: Multi-Head Attention ──────────────────────────────────────────────
# NOTE: dropout parameter added (defaults to 0.1 = original hardcoded value)
# so that Ray Tune can search over it without changing baseline behaviour.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear   = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear   = nn.Linear(d_model, d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.query_linear(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn   = self.dropout(torch.softmax(scores, dim=-1))
        out    = torch.matmul(attn, V)
        out    = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_linear(out)


# ── 0-F-3: FeedForward, LayerNorm, Encoder/Decoder Layers ───────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = LayerNorm(d_model)
        self.norm2     = LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = LayerNorm(d_model)
        self.norm2      = LayerNorm(d_model)
        self.norm3      = LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ── 0-F-4: Encoder, Decoder, Full Transformer ────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads,
                 d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers  = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads,
                 d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers  = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers,
                               num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers,
                               num_heads, d_ff, max_len, dropout)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)   # [B,1,1,T]

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones(size, size)).bool().to(
            next(self.parameters()).device)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask    = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_m   = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_m   = self.make_subsequent_mask(tgt.size(1))
        tgt_mask    = tgt_pad_m & tgt_sub_m

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)


# ── 0-G: Dataset & DataLoader ────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab     = en_vocab
        self.hi_vocab     = hi_vocab
        self.max_len      = max_len

    def __len__(self):  return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input  = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output


# ── 0-H: Translate & BLEU helpers ────────────────────────────────────────────
def translate_sentence(model, sentence, en_vocab, hi_vocab,
                       max_len=50, device=DEVICE):
    model.eval()
    tokens     = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_tokens = [hi_vocab["<sos>"]]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, SRC_PAD_IDX, TGT_PAD_IDX)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break

    return " ".join(hi_vocab.itos[i] for i in tgt_tokens[1:-1])


smoothie = SmoothingFunction().method4

def evaluate_bleu_nltk(model, dataset, en_vocab, hi_vocab, max_len=50):
    references, hypotheses = [], []
    for en_sent, hi_sent in dataset:
        pred        = translate_sentence(model, en_sent, en_vocab, hi_vocab, max_len)
        hypotheses.append(pred.split())
        references.append([hi_sent.split()])
    score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    print(f"BLEU Score (NLTK corpus_bleu): {score * 100:.2f}")
    return score


# Fixed validation set (same as original notebook)
val_dataset = [
    ("I love you.",                  "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?",                 "आप कैसे हैं?"),
    ("You should sleep.",            "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.",             "मुझे टॉम को बताने दीजिए।"),
]


# ---
# ## Part 1 — Baseline Training (100 epochs, hardcoded hyperparameters)
# 
# Run **exactly as-is** and record:
# - Total training time
# - Final epoch loss
# - BLEU score on `val_dataset`

# ── Part 1: Baseline Hyperparameters ─────────────────────────────────────────
MAX_LEN    = 50
BATCH_SIZE = 60
NUM_EPOCHS = 100
D_MODEL    = 512
NUM_LAYERS = 6
NUM_HEADS  = 8
D_FF       = 2048
DROPOUT    = 0.1
LR         = 1e-4


# ── Part 1: Build dataset & loader ───────────────────────────────────────────
dataset      = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)


# ── Part 1: Instantiate model, optimizer, criterion ─────────────────────────
baseline_model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    d_model=D_MODEL, num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS, d_ff=D_FF,
    max_len=MAX_LEN, dropout=DROPOUT
).to(DEVICE)

criterion        = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
baseline_optim   = optim.Adam(baseline_model.parameters(), lr=LR)


# ── Part 1: Checkpoint helpers (from original notebook) ──────────────────────
def save_checkpoint(epoch, model, optimizer, loss, path="checkpoint.pt"):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }, path)
    print(f"  Checkpoint saved — epoch {epoch}, loss {loss:.4f}")


def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
        return ckpt["epoch"]
    print("  No checkpoint found — starting from scratch.")
    return 0


# ── Part 1: Training loop (exact copy from en_to_hi.ipynb) ──────────────────
def train_baseline(model, loader, optimizer, criterion,
                   start_epoch=0, num_epochs=NUM_EPOCHS,
                   checkpoint_path="checkpoint.pt"):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for src, tgt_input, tgt_output in loop:
            src, tgt_input, tgt_output = (src.to(DEVICE),
                                          tgt_input.to(DEVICE),
                                          tgt_output.to(DEVICE))
            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        save_checkpoint(epoch + 1, model, optimizer, avg_loss, checkpoint_path)
    return avg_loss   # return last epoch average loss


# ── Part 1: SKIP RETRAINING — load from saved weights ────────────────────────
# Baseline already completed. Metrics recorded from previous run:
#   Time  : 3362.1 s  |  Loss : 0.0994  |  BLEU : 73.79
print("=" * 60)
print("  PART 1 — BASELINE (skipped, loading saved weights)")
print("=" * 60)

baseline_model.load_state_dict(
    torch.load("transformer_translation_final.pth", map_location=DEVICE)
)
baseline_model.eval()
print("  Loaded → transformer_translation_final.pth")

# Hardcoded from the completed baseline run
baseline_time        = 3362.1
final_baseline_loss  = 0.0994
baseline_bleu        = 0.7379   # 73.79 / 100

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║  BASELINE METRICS  (from prev run)   ║")
print(f"  ║  Time  :   3362.1 s                  ║")
print(f"  ║  Loss  :   0.0994                    ║")
print(f"  ║  BLEU  :   73.79 %                   ║")
print(f"  ╚══════════════════════════════════════╝")


# ---
# ## Part 2 — Ray Tune + Optuna Hyperparameter Search
# 
# ### Changes from baseline in `train_tune`:
# | Change | Reason |
# |--------|--------|
# | All HP from `config` dict | Ray Tune manages the values |
# | `ray.train.report()` instead of `save_checkpoint` | Ray Tune needs per-epoch metrics |
# | CosineAnnealingLR scheduler | Faster convergence than no scheduler in baseline |
# | `num_workers=0` in DataLoader | Required inside Ray workers |

# ── Part 2: train_tune() — Ray Tune compatible training function ─────────────
def train_tune(config):
    """
    Trains the Transformer with hyperparameters from `config`.
    Reports {'loss': avg_epoch_loss} to Ray Tune after every epoch.
    ASHA uses these reports to prune underperforming trials.
    """
    # ── Data ──
    _dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    _loader  = DataLoader(_dataset, batch_size=int(config["batch_size"]),
                          shuffle=True, collate_fn=collate_fn, num_workers=0)

    # ── Safety: d_model must be divisible by num_heads ──
    d_model   = int(config["d_model"])
    num_heads = int(config["num_heads"])
    while d_model % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    # ── Model ──
    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=d_model,
        num_layers=int(config["num_layers"]),
        num_heads=num_heads,
        d_ff=int(config["d_ff"]),
        max_len=MAX_LEN,
        dropout=config["dropout"],
    ).to(DEVICE)

    # ── Optimizer & Scheduler ──
    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config.get("weight_decay", 0.0))
    # CosineAnnealingLR reaches lower minima faster than a flat lr
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["num_epochs"]))
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

    # ── Epoch loop ──
    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        epoch_loss = 0
        for src, tgt_input, tgt_output in _loader:
            src, tgt_input, tgt_output = (src.to(DEVICE),
                                          tgt_input.to(DEVICE),
                                          tgt_output.to(DEVICE))
            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(_loader)

        # ── REQUIRED: report metrics to Ray Tune every epoch ──
        ray.train.report({"loss": avg_loss, "epoch": epoch})


# ### Search Space
# We tune **8 hyperparameters** (≥ 4 required by the rubric).
# 
# | # | HP | Range / Choices | API |
# |---|---|---|---|
# | 1 | `lr` | 1e-5 → 1e-3 | `loguniform` |
# | 2 | `batch_size` | 32, 64, 128 | `choice` |
# | 3 | `num_heads` | 4, 8 | `choice` |
# | 4 | `d_ff` | 1024, 2048, 4096 | `choice` |
# | 5 | `dropout` | 0.05 → 0.4 | `uniform` |
# | 6 | `d_model` | 256, 512 | `choice` |
# | 7 | `num_layers` | 3, 4, 6 | `choice` |
# | 8 | `weight_decay` | 1e-6 → 1e-3 | `loguniform` |
# 
# Each trial is **capped at 30 epochs** (< 100 baseline). ASHA prunes after epoch 5.

# ── Part 2: Search Space ──────────────────────────────────────────────────────
search_space = {
    "lr":           tune.loguniform(1e-5, 1e-3),      # 1. Learning rate
    "batch_size":   tune.choice([32, 64, 128]),        # 2. Batch size
    "num_heads":    tune.choice([4, 8]),               # 3. Attention heads
    "d_ff":         tune.choice([1024, 2048, 4096]),   # 4. FFN dimension
    "dropout":      tune.uniform(0.05, 0.4),            # 5. Dropout rate
    "d_model":      tune.choice([256, 512]),            # 6. Model dimension
    "num_layers":   tune.choice([3, 4, 6]),             # 7. Encoder/decoder layers
    "weight_decay": tune.loguniform(1e-6, 1e-3),       # 8. L2 regularisation

    # Efficiency challenge: cap each trial at 30 epochs (< 100 baseline)
    "num_epochs":   30,
}

print("Search space:")
for k, v in search_space.items():
    print(f"  {k:>15} : {v}")


# ── Part 2: ASHA Scheduler + Optuna ─────────────────────────────────────────
# ASHAScheduler: asynchronously stops bad trials early
asha = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=30,           # max epochs per trial
    grace_period=5,     # run at least 5 epochs before pruning
    reduction_factor=3, # keep top 1/3 of trials at each rung
)

# OptunaSearch: Bayesian TPE sampler — learns from past trials
optuna_search = OptunaSearch(metric="loss", mode="min")

print("ASHA    : max_t=30, grace_period=5, reduction_factor=3")
print("Optuna  : metric=loss, mode=min")


# ── Part 2: Run Sweep ─────────────────────────────────────────────────────────
print("=" * 60)
print("  PART 2 — RAY TUNE + OPTUNA SWEEP")
print("  20 trials × max 30 epochs each")
print("=" * 60)

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=False)

tuner = tune.Tuner(
    tune.with_resources(
        train_tune,
        resources={"cpu": 2, "gpu": 0.5 if torch.cuda.is_available() else 0}
    ),
    tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        scheduler=asha,
        num_samples=20,     # 20 different HP combinations
    ),
    param_space=search_space,
)

sweep_t0  = time.time()
results   = tuner.fit()
sweep_time = time.time() - sweep_t0
print(f"\nSweep completed in {sweep_time:.1f}s")


# ── Part 2: Extract Best Config ───────────────────────────────────────────────
best_result = results.get_best_result(metric="loss", mode="min")
best_config = best_result.config
best_loss   = best_result.metrics["loss"]

print("\n── Best Configuration ──────────────────────────────")
for k, v in best_config.items():
    print(f"  {k:>15} : {v}")
print(f"\n  Best trial loss : {best_loss:.4f}")


# ---
# ## Part 3 — Retrain Best Config & Evaluate BLEU
# 
# Retrain the model using the best hyperparameters found by Optuna.
# Evaluate BLEU every 5 epochs and check when/if it matches the baseline.

# ── Part 3: Build best model ─────────────────────────────────────────────────
d_model   = int(best_config["d_model"])
num_heads = int(best_config["num_heads"])
while d_model % num_heads != 0 and num_heads > 1:
    num_heads -= 1

best_model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    d_model=d_model,
    num_layers=int(best_config["num_layers"]),
    num_heads=num_heads,
    d_ff=int(best_config["d_ff"]),
    max_len=MAX_LEN,
    dropout=best_config["dropout"],
).to(DEVICE)

best_optim = optim.Adam(best_model.parameters(),
                        lr=best_config["lr"],
                        weight_decay=best_config.get("weight_decay", 0.0))
best_sched = optim.lr_scheduler.CosineAnnealingLR(best_optim, T_max=50)
best_crit  = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

best_loader = DataLoader(
    TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN),
    batch_size=int(best_config["batch_size"]),
    shuffle=True, collate_fn=collate_fn, num_workers=0
)

print(f"Model params: {sum(p.numel() for p in best_model.parameters()):,}")


# ── Part 3: Retrain & track BLEU ─────────────────────────────────────────────
print("=" * 60)
print("  PART 3 — RETRAIN BEST CONFIG  (up to 50 epochs)")
print("=" * 60)

beat_at_epoch = None
MAX_RETRAIN   = 50
t0            = time.time()

for epoch in range(1, MAX_RETRAIN + 1):
    best_model.train()
    epoch_loss = 0
    for src, tgt_input, tgt_output in best_loader:
        src, tgt_input, tgt_output = (src.to(DEVICE),
                                      tgt_input.to(DEVICE),
                                      tgt_output.to(DEVICE))
        output = best_model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
        output = output.view(-1, output.shape[-1])
        tgt_output = tgt_output.view(-1)

        loss = best_crit(output, tgt_output)
        best_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(best_model.parameters(), 1.0)
        best_optim.step()
        epoch_loss += loss.item()

    best_sched.step()
    avg_loss = epoch_loss / len(best_loader)

    # Evaluate BLEU every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        bleu = evaluate_bleu_nltk(best_model, val_dataset, en_vocab, hi_vocab)
        flag = "  ✓ BASELINE BEATEN!" if bleu >= baseline_bleu else ""
        print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | BLEU: {bleu*100:.2f}{flag}")
        if bleu >= baseline_bleu and beat_at_epoch is None:
            beat_at_epoch = epoch

retrain_time = time.time() - t0
final_bleu   = evaluate_bleu_nltk(best_model, val_dataset, en_vocab, hi_vocab)
final_loss   = avg_loss


# ── Save Best Model ───────────────────────────────────────────────────────────
torch.save(best_model.state_dict(), "rollno_ass_4_best_model.pth")
print("Best model saved → rollno_ass_4_best_model.pth")


# ## Final Comparison Report

# ── Final Comparison Report ───────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║                  FINAL COMPARISON                       ║")
print("╠══════════════════════════════════════╦══════════╦════════╣")
print(f"║ {'Metric':<37}║ {'Baseline':>8} ║ {'Best':>6} ║")
print("╠══════════════════════════════════════╬══════════╬════════╣")
print(f"║ {'Training Time (s)':<37}║ {baseline_time:>8.1f} ║ {retrain_time:>6.1f} ║")
print(f"║ {'Final Loss':<37}║ {final_baseline_loss:>8.4f} ║ {final_loss:>6.4f} ║")
print(f"║ {'BLEU Score':<37}║ {baseline_bleu*100:>7.2f}% ║ {final_bleu*100:>5.2f}% ║")
print(f"║ {'Epochs Used':<37}║ {'100':>8} ║ {MAX_RETRAIN:>6} ║")
print("╚══════════════════════════════════════╩══════════╩════════╝")

if beat_at_epoch:
    print(f"\n  ✓ Baseline BLEU matched/beaten at epoch {beat_at_epoch}"
          f"  ({beat_at_epoch}% of the 100-epoch baseline)")
else:
    print(f"\n  Best BLEU: {final_bleu*100:.2f}%  |  Baseline: {baseline_bleu*100:.2f}%")

print("\n  Best Hyperparameters Found:")
print(f"    lr           = {best_config['lr']:.2e}")
print(f"    batch_size   = {best_config['batch_size']}")
print(f"    num_heads    = {num_heads}")
print(f"    d_ff         = {best_config['d_ff']}")
print(f"    dropout      = {best_config['dropout']:.3f}")
print(f"    d_model      = {best_config['d_model']}")
print(f"    num_layers   = {best_config['num_layers']}")
print(f"    weight_decay = {best_config.get('weight_decay', 0):.2e}")

ray.shutdown()

