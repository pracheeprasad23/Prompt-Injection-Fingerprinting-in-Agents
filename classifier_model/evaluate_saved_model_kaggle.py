"""
Evaluate a saved prompt-injection classifier checkpoint on the test split.

Example:
python evaluate_saved_model_kaggle.py \
  --data /kaggle/working/cleaned_augmented_neuralchemy_dataset.jsonl \
  --model_dir /kaggle/working/saved_model/best_binary_injection \
  --output_dir /kaggle/working/saved_model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


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
        tool_strs = []
        for tc in tool_calls:
            tool_name = tc.get("tool", "unknown")
            tool_input = str(tc.get("input", ""))[:80]
            tool_strs.append(f"{tool_name}({tool_input})")
        parts.append(f"[TOOLS] {' | '.join(tool_strs)}")

    final_action = record.get("agent_final_action", "").strip()
    if final_action:
        parts.append(f"[ACTION] {final_action}")

    step = record.get("injection_detected_at_step", -1)
    step_str = str(step) if step != -1 else "none"
    parts.append(f"[STEP] {step_str}")

    return " ".join(parts)


def get_test_split(records: list) -> list:
    test = [r for r in records if str(r.get("split", "")).lower() == "test"]
    if not test:
        raise ValueError("No test split found in dataset (expected split='test').")
    return test


class EvalDataset(Dataset):
    def __init__(self, records, tokenizer, max_len: int):
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


def evaluate(model, loader, device, log_every: int = 10):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all, probs_all = [], [], []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        total_batches = len(loader)
        for step, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False), 1):
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

            if (log_every > 0 and step % log_every == 0) or step == total_batches:
                print(f"  [eval] step {step}/{total_batches} loss={(total_loss/step):.4f}")

    avg_loss = total_loss / max(1, len(loader))
    f1_w = f1_score(labels_all, preds_all, average="weighted")
    f1_macro = f1_score(labels_all, preds_all, average="macro")
    f1_mal = f1_score(labels_all, preds_all, average="binary", pos_label=1)

    try:
        auc = roc_auc_score(labels_all, [p[1] for p in probs_all])
    except Exception:
        auc = None

    return avg_loss, f1_w, f1_macro, f1_mal, auc, preds_all, labels_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved prompt-injection model on test split")
    parser.add_argument("--data", required=True, help="Path to cleaned JSONL dataset")
    parser.add_argument("--model_dir", required=True, help="Saved model directory")
    parser.add_argument("--output_dir", default=".", help="Directory for outputs")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Eval device: {device}")

    print(f"[INFO] Loading dataset: {args.data}")
    rows = load_jsonl(args.data)
    test_rows = get_test_split(rows)
    print(f"[INFO] Test rows: {len(test_rows)}")

    print(f"[INFO] Loading tokenizer from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=args.use_fast_tokenizer)
    print(f"[INFO] Loading model from: {args.model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        use_safetensors=True,
    ).to(device)

    ds = EvalDataset(test_rows, tokenizer, args.maxlen)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if device.type == "cuda" else None)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=collator)
    print(f"[INFO] Eval DataLoader: batches={len(loader)}, batch_size={args.batch}, num_workers=0")

    test_loss, test_f1_w, test_f1_macro, test_f1_mal, test_auc, preds, labels = evaluate(
        model,
        loader,
        device,
        log_every=args.log_every,
    )

    cm = confusion_matrix(labels, preds)
    report_text = classification_report(labels, preds, target_names=["benign", "malicious"], digits=4)

    print("\nTEST RESULTS")
    print(f"Loss         : {test_loss:.4f}")
    print(f"F1_weighted  : {test_f1_w:.4f}")
    print(f"F1_macro     : {test_f1_macro:.4f}")
    print(f"F1_malicious : {test_f1_mal:.4f}")
    if test_auc is not None:
        print(f"AUC          : {test_auc:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "test_loss": test_loss,
        "test_f1_weighted": test_f1_w,
        "test_f1_macro": test_f1_macro,
        "test_f1_malicious": test_f1_mal,
        "test_auc": test_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=["benign", "malicious"],
            digits=4,
            output_dict=True,
        ),
        "test_size": len(test_rows),
    }

    with open(out_dir / "standalone_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "standalone_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved: {out_dir / 'standalone_test_results.json'}")
    print(f"Saved: {out_dir / 'standalone_test_report.txt'}")


if __name__ == "__main__":
    main()
