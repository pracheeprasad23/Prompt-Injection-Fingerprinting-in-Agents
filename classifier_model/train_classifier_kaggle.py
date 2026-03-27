"""
Single-process Kaggle trainer for prompt-injection classification.

Behavior:
- Uses split field strictly: train for optimization, val for model selection, test for final evaluation.
- Runs one training job only (no fallback orchestration).
- Uses one GPU device (cuda:0) when available.
- Uses DataLoader with num_workers=0 to avoid worker subprocesses.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "microsoft/deberta-v3-small"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_input_text(record: dict) -> str:
    parts = []

    text = record.get("text", record.get("prompt", "")).strip()
    if text:
        parts.append(f"[PROMPT] {text}")

    trace = record.get("reasoning_trace", [])
    if trace:
        thoughts = " | ".join(str(t).strip() for t in trace if t)
        if thoughts:
            parts.append(f"[THOUGHTS] {thoughts}")

    tool_calls = record.get("tool_calls", [])
    if tool_calls:
        tc_parts = []
        for tc in tool_calls:
            tool_name = tc.get("tool", "unknown")
            tool_input = str(tc.get("input", ""))[:80]
            tc_parts.append(f"{tool_name}({tool_input})")
        parts.append(f"[TOOLS] {' | '.join(tc_parts)}")

    final_action = record.get("agent_final_action", "").strip()
    if final_action:
        parts.append(f"[ACTION] {final_action}")

    step = record.get("injection_detected_at_step", -1)
    parts.append(f"[STEP] {step if step != -1 else 'none'}")

    return " ".join(parts)


def split_from_field(records: list):
    train = [r for r in records if str(r.get("split", "")).lower() == "train"]
    val = [r for r in records if str(r.get("split", "")).lower() in {"validation", "val", "valid"}]
    test = [r for r in records if str(r.get("split", "")).lower() == "test"]
    return train, val, test


def compute_class_weights(train_records: list, device: torch.device) -> torch.Tensor:
    counts = Counter(int(r["label"]) for r in train_records)
    total = sum(counts.values())
    weights = [total / (2 * counts.get(i, 1)) for i in range(2)]
    return torch.tensor(weights, dtype=torch.float, device=device)


class InjectionDataset(Dataset):
    def __init__(self, records: list, tokenizer, max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            build_input_text(rec),
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = int(rec["label"])
        return enc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            weight = criterion.weight
            if weight is not None:
                weight = weight.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, labels, weight=weight)

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    f1_w = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_mal = f1_score(all_labels, all_preds, average="binary", pos_label=1)

    try:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except Exception:
        auc = None

    return avg_loss, f1_w, f1_macro, f1_mal, auc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Single-process Kaggle trainer")
    parser.add_argument("--data", required=True, help="Path to cleaned JSONL dataset")
    parser.add_argument("--output_dir", default="/kaggle/working/saved_model", help="Output directory")
    parser.add_argument("--model_name", default=MODEL_NAME, help="HF model name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] Visible GPUs: {torch.cuda.device_count()} (using only cuda:0)")

    rows = load_jsonl(args.data)
    print(f"[INFO] Loaded rows: {len(rows)}")

    train_records, val_records, test_records = split_from_field(rows)
    if not train_records or not val_records or not test_records:
        raise RuntimeError(
            "Missing required split rows. Expected split values: train, validation/val, test."
        )

    print(f"[INFO] Split sizes -> train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    print("[INFO] Training uses only train split rows.")
    print("[INFO] Validation and test are evaluation-only.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = InjectionDataset(train_records, tokenizer, args.maxlen)
    val_ds = InjectionDataset(val_records, tokenizer, args.maxlen)
    test_ds = InjectionDataset(test_records, tokenizer, args.maxlen)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if device.type == "cuda" else None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
        dtype=torch.float32,
    )

    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = args.dropout
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = args.dropout
    if hasattr(model.config, "classifier_dropout"):
        model.config.classifier_dropout = args.dropout

    model = model.to(device).float()

    class_weights = compute_class_weights(train_records, device)
    print(f"[INFO] Class weights: {class_weights.detach().cpu().numpy().round(3)}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best_binary_injection"

    best_score = -1.0
    best_epoch = -1
    patience_counter = 0

    print("\n" + "=" * 65)
    print("Training: binary_injection")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.time()

        print(f"\n[INFO] Starting epoch {epoch}/{args.epochs} ...")

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, 1):
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            weight = class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = F.cross_entropy(logits, labels, weight=weight, label_smoothing=args.label_smoothing)
            loss = loss / args.grad_accum

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)
                running_loss += loss.item() * args.grad_accum

            if args.log_every > 0 and (step % args.log_every == 0 or step == len(train_loader)):
                elapsed = time.time() - epoch_start
                print(
                    f"  [epoch {epoch}] step {step}/{len(train_loader)} "
                    f"loss={(running_loss/max(1, step)):.4f} "
                    f"acc={(running_correct/max(1, running_total)):.4f} "
                    f"elapsed={elapsed/60:.1f}m"
                )

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = running_correct / max(1, running_total)

        val_loss, val_f1_w, val_f1_macro, val_f1_mal, val_auc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        auc_text = f"  AUC={val_auc:.4f}" if val_auc is not None else ""
        print(
            f"Epoch {epoch}: Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} F1_w={val_f1_w:.4f} F1_macro={val_f1_macro:.4f} "
            f"F1_mal={val_f1_mal:.4f}{auc_text}"
        )

        if np.isfinite(val_loss) and np.isfinite(val_f1_mal) and val_f1_mal > best_score:
            best_score = val_f1_mal
            best_epoch = epoch
            patience_counter = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            print(f"  Saved new best checkpoint at epoch {epoch} (val F1_mal={val_f1_mal:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Early-stop counter: {patience_counter}/{args.patience}")

        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print("\n[INFO] Early stopping triggered.")
            break

    if best_epoch == -1:
        raise RuntimeError("Training ended without a valid checkpoint.")

    print("\n[INFO] Loading best checkpoint for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(best_dir),
        num_labels=2,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    ).to(device)

    test_loss, test_f1_w, test_f1_macro, test_f1_mal, test_auc, test_preds, test_labels = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
    )

    print("\n" + "-" * 65)
    print("TEST RESULTS")
    print("-" * 65)
    print(f"Loss         : {test_loss:.4f}")
    print(f"F1_weighted  : {test_f1_w:.4f}")
    print(f"F1_macro     : {test_f1_macro:.4f}")
    print(f"F1_malicious : {test_f1_mal:.4f}")
    if test_auc is not None:
        print(f"AUC          : {test_auc:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    report_text = classification_report(test_labels, test_preds, target_names=["benign", "malicious"], digits=4)

    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    results = {
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "best_val_f1_malicious": best_score,
        "test_loss": test_loss,
        "test_f1_weighted": test_f1_w,
        "test_f1_macro": test_f1_macro,
        "test_f1_malicious": test_f1_mal,
        "test_auc": test_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            test_labels,
            test_preds,
            target_names=["benign", "malicious"],
            digits=4,
            output_dict=True,
        ),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "test_size": len(test_records),
        "effective_batch_size": args.batch * args.grad_accum,
        "device": str(device),
    }

    results_path = out_dir / "results_binary_injection_kaggle.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "classification_report_binary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[INFO] Best model saved to: {best_dir}")
    print(f"[INFO] Results saved to   : {results_path}")
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
