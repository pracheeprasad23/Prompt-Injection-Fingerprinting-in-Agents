"""
Phase 4 — Full Evaluation: Baselines, Ablations, Latency, Plots
=================================================================
Run this AFTER train_classifier.py has completed.

What this produces:
  1. Baseline comparisons  (TF-IDF + LR, prompt-only DeBERTa, full model)
  2. Ablation study        (remove one signal at a time)
  3. Latency benchmark     (ms per sample, throughput)
  4. Plots                 (confusion matrix, ROC curve, F1 bar chart, ablation bar chart)
  5. results_phase4.json   (all numbers in one file, ready for paper)

Requirements:
    pip install transformers torch scikit-learn tqdm numpy matplotlib seaborn

Usage:
    python evaluate_baselines.py \
        --data  cleaned_augmented_neuralchemy_dataset.jsonl \
        --model ./saved_model/best_binary_injection \
        --output_dir ./phase4_results
"""

import json
import argparse
import os
import time
import random
import numpy as np
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_fscore_support
)
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — works without display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    PLOTTING = True
except ImportError:
    PLOTTING = False
    print("[WARN] matplotlib/seaborn not found — skipping plots. Run: pip install matplotlib seaborn")


MODEL_NAME = "microsoft/deberta-v3-small"
SEED = 42


# ─────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except:
                pass
    return records


def prepare_splits(records, seed=42):
    """
    Prepare dataset splits using the 'split' field if available.
    Falls back to random split if field is missing.
    """
    # Check if records have a 'split' field
    has_split_field = any(r.get("split") for r in records)
    
    if has_split_field:
        # Use the 'split' field (RECOMMENDED for pre-split datasets)
        train = [r for r in records if str(r.get("split", "")).lower() == "train"]
        val = [r for r in records if str(r.get("split", "")).lower() in {"validation", "val", "valid"}]
        test = [r for r in records if str(r.get("split", "")).lower() == "test"]
        
        if train and val and test:
            print(f"[INFO] Using 'split' field: train={len(train)}, val={len(val)}, test={len(test)}")
            return train, val, test
        else:
            print(f"[WARN] Split field incomplete (train={len(train)}, val={len(val)}, test={len(test)}). Falling back to random split.")
    
    # Fallback: Random split
    labels = [int(r["label"]) for r in records]
    train_val, test = train_test_split(records, test_size=0.15, stratify=labels, random_state=seed)
    labels_tv = [int(r["label"]) for r in train_val]
    train, val = train_test_split(train_val, test_size=0.15/0.85, stratify=labels_tv, random_state=seed)
    print(f"[WARN] Using random split (seed={seed}): train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def compute_class_weights(records, num_classes, device):
    counts = Counter(int(r["label"]) for r in records)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float).to(device)


# ─────────────────────────────────────────────────────────
# FEATURE BUILDERS — one per ablation variant
# ─────────────────────────────────────────────────────────
def get_prompt_only(record):
    text = record.get("text", record.get("prompt", "")).strip()
    return f"[PROMPT] {text}"


def get_full_input(record):
    parts = []
    text = record.get("text", record.get("prompt", "")).strip()
    if text:
        parts.append(f"[PROMPT] {text}")
    trace = record.get("reasoning_trace", [])
    if trace:
        parts.append(f"[THOUGHTS] {' | '.join(str(t).strip() for t in trace if t)}")
    tool_calls = record.get("tool_calls", [])
    if tool_calls:
        tool_strs = [f"{tc.get('tool','?')}({str(tc.get('input',''))[:80]})" for tc in tool_calls]
        parts.append(f"[TOOLS] {' | '.join(tool_strs)}")
    final_action = record.get("agent_final_action", "").strip()
    if final_action:
        parts.append(f"[ACTION] {final_action}")
    step = record.get("injection_detected_at_step", -1)
    parts.append(f"[STEP] {'none' if step == -1 else step}")
    return " ".join(parts)


def get_no_thoughts(record):
    """Ablation: remove reasoning trace"""
    parts = []
    text = record.get("text", record.get("prompt", "")).strip()
    if text:
        parts.append(f"[PROMPT] {text}")
    tool_calls = record.get("tool_calls", [])
    if tool_calls:
        tool_strs = [f"{tc.get('tool','?')}({str(tc.get('input',''))[:80]})" for tc in tool_calls]
        parts.append(f"[TOOLS] {' | '.join(tool_strs)}")
    final_action = record.get("agent_final_action", "").strip()
    if final_action:
        parts.append(f"[ACTION] {final_action}")
    step = record.get("injection_detected_at_step", -1)
    parts.append(f"[STEP] {'none' if step == -1 else step}")
    return " ".join(parts)


def get_no_tools(record):
    """Ablation: remove tool calls"""
    parts = []
    text = record.get("text", record.get("prompt", "")).strip()
    if text:
        parts.append(f"[PROMPT] {text}")
    trace = record.get("reasoning_trace", [])
    if trace:
        parts.append(f"[THOUGHTS] {' | '.join(str(t).strip() for t in trace if t)}")
    final_action = record.get("agent_final_action", "").strip()
    if final_action:
        parts.append(f"[ACTION] {final_action}")
    step = record.get("injection_detected_at_step", -1)
    parts.append(f"[STEP] {'none' if step == -1 else step}")
    return " ".join(parts)


def get_no_step(record):
    """Ablation: remove injection step signal"""
    parts = []
    text = record.get("text", record.get("prompt", "")).strip()
    if text:
        parts.append(f"[PROMPT] {text}")
    trace = record.get("reasoning_trace", [])
    if trace:
        parts.append(f"[THOUGHTS] {' | '.join(str(t).strip() for t in trace if t)}")
    tool_calls = record.get("tool_calls", [])
    if tool_calls:
        tool_strs = [f"{tc.get('tool','?')}({str(tc.get('input',''))[:80]})" for tc in tool_calls]
        parts.append(f"[TOOLS] {' | '.join(tool_strs)}")
    final_action = record.get("agent_final_action", "").strip()
    if final_action:
        parts.append(f"[ACTION] {final_action}")
    return " ".join(parts)


# Map variant names to their builder functions
ABLATION_VARIANTS = {
    "Full Model (Ours)":      get_full_input,
    "w/o Reasoning Trace":    get_no_thoughts,
    "w/o Tool Calls":         get_no_tools,
    "w/o Injection Step":     get_no_step,
    "Prompt Only (DeBERTa)":  get_prompt_only,
}


# ─────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────
class InjectionDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, feature_fn):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.feature_fn = feature_fn

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        text  = self.feature_fn(rec)
        label = int(rec["label"])
        enc = self.tokenizer(
            text, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# DEBERTA EVALUATION
# ─────────────────────────────────────────────────────────
def evaluate_deberta(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="    Evaluating", leave=False):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)
            all_preds .extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs .extend(probs.cpu().numpy())
    return all_preds, all_labels, all_probs


def fine_tune_deberta(train_records, val_records, tokenizer, feature_fn,
                       device, epochs=3, batch_size=16, lr=2e-5, max_len=256):
    """Quick fine-tune for ablation variants — 3 epochs each."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device).float()

    train_ds = InjectionDataset(train_records, tokenizer, max_len, feature_fn)
    val_ds   = InjectionDataset(val_records,   tokenizer, max_len, feature_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_records, 2, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"    Epoch {epoch+1}/{epochs}", leave=False):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids=ids, attention_mask=mask).logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        preds, lbls, _ = evaluate_deberta(model, val_loader, device)
        f1 = f1_score(lbls, preds, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────
# LATENCY BENCHMARK
# ─────────────────────────────────────────────────────────
def measure_latency(model, tokenizer, records, device, feature_fn, n_samples=100, max_len=256):
    model.eval()
    samples = random.sample(records, min(n_samples, len(records)))
    latencies = []
    with torch.no_grad():
        for rec in samples:
            text = feature_fn(rec)
            enc  = tokenizer(text, max_length=max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            # Warmup
            _ = model(input_ids=ids, attention_mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed run
            start = time.perf_counter()
            _ = model(input_ids=ids, attention_mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # ms

    return {
        "mean_ms":   round(float(np.mean(latencies)), 3),
        "median_ms": round(float(np.median(latencies)), 3),
        "p95_ms":    round(float(np.percentile(latencies, 95)), 3),
        "p99_ms":    round(float(np.percentile(latencies, 99)), 3),
        "throughput_per_sec": round(1000.0 / float(np.mean(latencies)), 1),
    }


# ─────────────────────────────────────────────────────────
# BASELINE 1: TF-IDF + Logistic Regression (prompt text only)
# ─────────────────────────────────────────────────────────
def run_tfidf_baseline(train_records, test_records):
    print("\n  [Baseline] TF-IDF + Logistic Regression (prompt only)...")
    train_texts  = [r.get("text", r.get("prompt", "")) for r in train_records]
    test_texts   = [r.get("text", r.get("prompt", "")) for r in test_records]
    train_labels = [int(r["label"]) for r in train_records]
    test_labels  = [int(r["label"]) for r in test_records]

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED, C=1.0
    )
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    f1  = f1_score(test_labels, preds, average="weighted")
    auc = roc_auc_score(test_labels, probs)
    p, r, f, _ = precision_recall_fscore_support(test_labels, preds, average="weighted")
    cm  = confusion_matrix(test_labels, preds)

    print(f"    F1={f1:.4f}  AUC={auc:.4f}")
    return {
        "name":      "TF-IDF + LR (Prompt Only)",
        "f1":        round(f1, 4),
        "auc":       round(auc, 4),
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            test_labels, preds,
            target_names=["benign", "malicious"],
            digits=4, output_dict=True
        ),
    }, test_labels, probs


# ─────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_roc_curves(roc_data, save_path):
    """roc_data: list of (name, fpr, tpr, auc)"""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
    for i, (name, fpr, tpr, auc) in enumerate(roc_data):
        ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                label=f"{name} (AUC={auc:.4f})")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Baseline Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_f1_bars(names, f1_scores, title, save_path, highlight_idx=0):
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5))
    colors = ["#2563eb" if i == highlight_idx else "#94a3b8" for i in range(len(names))]
    bars = ax.bar(names, f1_scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_ylim(max(0, min(f1_scores) - 0.05), min(1.0, max(f1_scores) + 0.08))
    ax.set_ylabel("Weighted F1 Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    highlight = mpatches.Patch(color="#2563eb", label="Proposed Method")
    ax.legend(handles=[highlight], fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_latency(latency_stats, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    keys   = ["mean_ms", "median_ms", "p95_ms", "p99_ms"]
    labels = ["Mean", "Median", "P95", "P99"]
    values = [latency_stats[k] for k in keys]
    bars = ax.bar(labels, values, color=["#2563eb", "#3b82f6", "#f59e0b", "#ef4444"],
                  edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(
        f"Inference Latency per Sample\n(Throughput: {latency_stats['throughput_per_sec']} samples/sec)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,  help="Path to JSONL dataset")
    parser.add_argument("--model",      required=True,  help="Path to saved Phase 3 model dir")
    parser.add_argument("--output_dir", default="./phase4_results")
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--maxlen",     type=int, default=256)
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Epochs for ablation re-training (default: 3, less than Phase 3)")
    args = parser.parse_args()

    # ── Validate input paths ──
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    
    if not os.path.isdir(args.model):
        raise FileNotFoundError(f"Model directory not found: {args.model}")
    
    # Check for required model files
    required_model_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    for fname in required_model_files:
        fpath = os.path.join(args.model, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing '{fname}' in model directory: {args.model}")

    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device : {device}")
    print(f"[INFO] Model  : {args.model}")
    print(f"[INFO] Output : {args.output_dir}")
    print(f"[INFO] Data   : {args.data}\n")

    # ── Load data + splits ────────────────────────────────
    print("[INFO] Loading dataset...")
    records = load_jsonl(args.data)
    print(f"[INFO] {len(records)} records loaded")
    train_records, val_records, test_records = prepare_splits(records, SEED)
    print(f"[INFO] Train={len(train_records)}  Val={len(val_records)}  Test={len(test_records)}")

    # ── Load tokenizer ────────────────────────────────────
    print(f"\n[INFO] Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Load Phase 3 best model ───────────────────────────
    print(f"[INFO] Loading Phase 3 model...")
    full_model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).float()
    full_model.eval()

    all_results = {}
    roc_data    = []

    # ══════════════════════════════════════════════════════
    # SECTION 1: BASELINES
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print("  SECTION 1: BASELINE COMPARISONS")
    print(f"{'='*55}")

    # Baseline 1 — TF-IDF + LR
    tfidf_result, test_labels, tfidf_probs = run_tfidf_baseline(train_records, test_records)
    all_results["baseline_tfidf_lr"] = tfidf_result
    fpr, tpr, _ = roc_curve(test_labels, tfidf_probs)
    roc_data.append(("TF-IDF + LR", fpr, tpr, tfidf_result["auc"]))

    # Baseline 2 — DeBERTa, prompt only (fine-tune fresh)
    print("\n  [Baseline] DeBERTa, prompt only (fine-tuning 3 epochs)...")
    prompt_only_model = fine_tune_deberta(
        train_records, val_records, tokenizer, get_prompt_only,
        device, epochs=args.epochs, batch_size=args.batch, max_len=args.maxlen
    )
    test_ds_prompt = InjectionDataset(test_records, tokenizer, args.maxlen, get_prompt_only)
    test_loader_prompt = DataLoader(test_ds_prompt, batch_size=args.batch, shuffle=False, num_workers=0)
    p_preds, p_labels, p_probs = evaluate_deberta(prompt_only_model, test_loader_prompt, device)
    p_f1  = f1_score(p_labels, p_preds, average="weighted")
    p_probs_pos = [x[1] for x in p_probs]
    p_auc = roc_auc_score(p_labels, p_probs_pos)
    prec, rec, _, _ = precision_recall_fscore_support(p_labels, p_preds, average="weighted")
    p_cm  = confusion_matrix(p_labels, p_preds)
    print(f"    F1={p_f1:.4f}  AUC={p_auc:.4f}")
    baseline_prompt = {
        "name": "DeBERTa (Prompt Only)",
        "f1": round(p_f1, 4), "auc": round(p_auc, 4),
        "precision": round(float(prec), 4), "recall": round(float(rec), 4),
        "confusion_matrix": p_cm.tolist(),
        "report": classification_report(p_labels, p_preds,
            target_names=["benign", "malicious"], digits=4, output_dict=True),
    }
    all_results["baseline_deberta_prompt_only"] = baseline_prompt
    fpr, tpr, _ = roc_curve(p_labels, p_probs_pos)
    roc_data.append(("DeBERTa Prompt Only", fpr, tpr, round(p_auc, 4)))

    # Full model evaluation on test set
    print("\n  [Full Model] Evaluating Phase 3 model on test set...")
    test_ds_full = InjectionDataset(test_records, tokenizer, args.maxlen, get_full_input)
    test_loader_full = DataLoader(test_ds_full, batch_size=args.batch, shuffle=False, num_workers=0)
    f_preds, f_labels, f_probs = evaluate_deberta(full_model, test_loader_full, device)
    f_f1  = f1_score(f_labels, f_preds, average="weighted")
    f_probs_pos = [x[1] for x in f_probs]
    f_auc = roc_auc_score(f_labels, f_probs_pos)
    prec, rec, _, _ = precision_recall_fscore_support(f_labels, f_preds, average="weighted")
    f_cm  = confusion_matrix(f_labels, f_preds)
    print(f"    F1={f_f1:.4f}  AUC={f_auc:.4f}")
    full_result = {
        "name": "Full Model — Ours (DeBERTa + Trace + Tools + Step)",
        "f1": round(f_f1, 4), "auc": round(f_auc, 4),
        "precision": round(float(prec), 4), "recall": round(float(rec), 4),
        "confusion_matrix": f_cm.tolist(),
        "report": classification_report(f_labels, f_preds,
            target_names=["benign", "malicious"], digits=4, output_dict=True),
    }
    all_results["full_model"] = full_result
    fpr, tpr, _ = roc_curve(f_labels, f_probs_pos)
    roc_data.append(("Full Model (Ours)", fpr, tpr, round(f_auc, 4)))

    # ══════════════════════════════════════════════════════
    # SECTION 2: ABLATION STUDY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print("  SECTION 2: ABLATION STUDY")
    print(f"{'='*55}")

    ablation_results = {}
    ablation_names   = []
    ablation_f1s     = []

    ablation_variants = {
        "Full Model (Ours)":   get_full_input,
        "w/o Reasoning Trace": get_no_thoughts,
        "w/o Tool Calls":      get_no_tools,
        "w/o Injection Step":  get_no_step,
    }

    for variant_name, feature_fn in ablation_variants.items():
        print(f"\n  Ablation: {variant_name}")
        if variant_name == "Full Model (Ours)":
            # Already evaluated above
            v_f1  = f_f1
            v_auc = f_auc
            v_preds, v_labels = f_preds, f_labels
        else:
            abl_model = fine_tune_deberta(
                train_records, val_records, tokenizer, feature_fn,
                device, epochs=args.epochs, batch_size=args.batch, max_len=args.maxlen
            )
            test_ds_abl = InjectionDataset(test_records, tokenizer, args.maxlen, feature_fn)
            test_loader_abl = DataLoader(test_ds_abl, batch_size=args.batch, shuffle=False, num_workers=0)
            v_preds, v_labels, v_probs_raw = evaluate_deberta(abl_model, test_loader_abl, device)
            v_f1  = f1_score(v_labels, v_preds, average="weighted")
            v_auc = roc_auc_score(v_labels, [x[1] for x in v_probs_raw])

        print(f"    F1={v_f1:.4f}  AUC={v_auc:.4f}")
        ablation_results[variant_name] = {"f1": round(v_f1, 4), "auc": round(v_auc, 4)}
        ablation_names.append(variant_name)
        ablation_f1s.append(round(v_f1, 4))

    all_results["ablation_study"] = ablation_results

    # ══════════════════════════════════════════════════════
    # SECTION 3: LATENCY BENCHMARK
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print("  SECTION 3: LATENCY BENCHMARK")
    print(f"{'='*55}")
    print("  Measuring inference latency (100 samples)...")
    latency = measure_latency(
        full_model, tokenizer, test_records, device, get_full_input, n_samples=100
    )
    print(f"  Mean    : {latency['mean_ms']} ms")
    print(f"  Median  : {latency['median_ms']} ms")
    print(f"  P95     : {latency['p95_ms']} ms")
    print(f"  P99     : {latency['p99_ms']} ms")
    print(f"  Throughput : {latency['throughput_per_sec']} samples/sec")
    all_results["latency"] = latency

    # ══════════════════════════════════════════════════════
    # SECTION 4: PLOTS
    # ══════════════════════════════════════════════════════
    if PLOTTING:
        print(f"\n{'='*55}")
        print("  SECTION 4: GENERATING PLOTS")
        print(f"{'='*55}")

        # Confusion matrix — Full model
        plot_confusion_matrix(
            np.array(f_cm),
            "Confusion Matrix — Full Model (Ours)",
            os.path.join(args.output_dir, "confusion_matrix_full_model.png")
        )

        # Confusion matrix — prompt only baseline
        plot_confusion_matrix(
            np.array(p_cm),
            "Confusion Matrix — DeBERTa Prompt Only",
            os.path.join(args.output_dir, "confusion_matrix_prompt_only.png")
        )

        # ROC curves
        plot_roc_curves(
            roc_data,
            os.path.join(args.output_dir, "roc_curves.png")
        )

        # Baseline F1 bar chart
        baseline_names = ["TF-IDF + LR", "DeBERTa\n(Prompt Only)", "Full Model\n(Ours)"]
        baseline_f1s   = [tfidf_result["f1"], baseline_prompt["f1"], full_result["f1"]]
        plot_f1_bars(
            baseline_names, baseline_f1s,
            "Baseline Comparison — Weighted F1",
            os.path.join(args.output_dir, "f1_baseline_comparison.png"),
            highlight_idx=2
        )

        # Ablation F1 bar chart
        plot_f1_bars(
            ablation_names, ablation_f1s,
            "Ablation Study — Weighted F1",
            os.path.join(args.output_dir, "f1_ablation_study.png"),
            highlight_idx=0
        )

        # Latency chart
        plot_latency(
            latency,
            os.path.join(args.output_dir, "latency_benchmark.png")
        )
    else:
        print("\n[WARN] Skipping plots — matplotlib not installed.")

    # ══════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════
    results_path = os.path.join(args.output_dir, "results_phase4.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ══════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Method':<40} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*56}")
    print(f"  {'TF-IDF + LR (Prompt Only)':<40} {tfidf_result['f1']:>8.4f} {tfidf_result['auc']:>8.4f}")
    print(f"  {'DeBERTa (Prompt Only)':<40} {baseline_prompt['f1']:>8.4f} {baseline_prompt['auc']:>8.4f}")
    print(f"  {'Full Model — Ours':<40} {full_result['f1']:>8.4f} {full_result['auc']:>8.4f}")
    print(f"\n  Ablation Study:")
    print(f"  {'Variant':<40} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*56}")
    for name, res in ablation_results.items():
        print(f"  {name:<40} {res['f1']:>8.4f} {res['auc']:>8.4f}")
    print(f"\n  Latency: {latency['mean_ms']}ms mean | {latency['throughput_per_sec']} samples/sec")
    print(f"\n  All results saved to : {results_path}")
    if PLOTTING:
        print(f"  Plots saved to       : {args.output_dir}/*.png")
    print(f"\n✅ Phase 4 complete!\n")


if __name__ == "__main__":
    main()
