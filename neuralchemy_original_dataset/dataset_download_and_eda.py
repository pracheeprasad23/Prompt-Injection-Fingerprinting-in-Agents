"""
Dataset Download & EDA Script
==============================
Downloads the neuralchemy/Prompt-injection-dataset (core split) from Hugging Face,
removes unnecessary columns, drops duplicates, and generates distribution insights.

Usage (Kaggle):
    1. Upload this script to Kaggle or paste into a notebook cell
    2. Make sure internet is enabled in notebook settings
    3. Run: python dataset_download_and_eda.py

Requirements:
    pip install datasets pandas matplotlib seaborn
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ─────────────────────────────────────────────
# 1. Download dataset from Hugging Face
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Downloading neuralchemy/Prompt-injection-dataset (core)")
print("=" * 60)

from datasets import load_dataset

ds = load_dataset("neuralchemy/Prompt-injection-dataset", "core")

print(f"\nDataset loaded successfully!")
print(f"  Train split : {len(ds['train'])} rows")
print(f"  Val split   : {len(ds['validation'])} rows")
print(f"  Test split  : {len(ds['test'])} rows")
print(f"  Total       : {len(ds['train']) + len(ds['validation']) + len(ds['test'])} rows")

# Combine all splits for EDA
train_df = ds["train"].to_pandas()
val_df   = ds["validation"].to_pandas()
test_df  = ds["test"].to_pandas()

# Tag each row with its split
train_df["split"] = "train"
val_df["split"]   = "validation"
test_df["split"]  = "test"

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"\nCombined DataFrame shape: {full_df.shape}")
print(f"Columns: {list(full_df.columns)}")

# ─────────────────────────────────────────────
# 2. Inspect raw data
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Raw Data Inspection")
print("=" * 60)

print("\n--- First 3 rows ---")
print(full_df.head(3).to_string())

print("\n--- Data Types ---")
print(full_df.dtypes)

print("\n--- Null Values ---")
print(full_df.isnull().sum())

# Check what 'tags' column looks like — user said it's always []
print("\n--- Tags column sample values ---")
print(full_df["tags"].value_counts().head(10))

# Check 'augmented' column — should be all False for core split
print("\n--- Augmented column distribution ---")
print(full_df["augmented"].value_counts())

# Check 'source' column
print("\n--- Source column distribution ---")
print(full_df["source"].value_counts())

# Check 'group_id' column — sample values
print("\n--- Group ID sample values ---")
print(full_df["group_id"].head(10))

# ─────────────────────────────────────────────
# 3. Remove unnecessary columns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Removing Unnecessary Columns")
print("=" * 60)

cols_to_remove = ["augmented", "group_id", "source", "tags"]
print(f"Removing columns: {cols_to_remove}")

for col in cols_to_remove:
    if col in full_df.columns:
        full_df.drop(columns=[col], inplace=True)

print(f"Remaining columns: {list(full_df.columns)}")
print(f"Shape after column removal: {full_df.shape}")

# ─────────────────────────────────────────────
# 4. Duplicate Detection & Removal
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Duplicate Detection & Removal")
print("=" * 60)

# Check for exact duplicates on 'text' column
text_duplicates = full_df.duplicated(subset=["text"], keep=False)
num_text_duplicates = text_duplicates.sum()
print(f"\nExact duplicate rows (by 'text' column): {num_text_duplicates}")

if num_text_duplicates > 0:
    print("\n--- Sample duplicates ---")
    dup_texts = full_df[text_duplicates].sort_values("text")
    print(dup_texts[["text", "label", "category", "split"]].head(20).to_string())

    # Check for CROSS-SPLIT duplicates (same text in train AND val/test = data leakage!)
    train_texts = set(full_df[full_df["split"] == "train"]["text"].values)
    val_texts   = set(full_df[full_df["split"] == "validation"]["text"].values)
    test_texts  = set(full_df[full_df["split"] == "test"]["text"].values)

    train_val_overlap  = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap   = val_texts & test_texts

    print(f"\n--- Cross-split text overlaps (DATA LEAKAGE CHECK) ---")
    print(f"  Train ∩ Validation : {len(train_val_overlap)} texts")
    print(f"  Train ∩ Test       : {len(train_test_overlap)} texts")
    print(f"  Validation ∩ Test  : {len(val_test_overlap)} texts")

# Remove duplicates — keep first occurrence
before_dedup = len(full_df)
full_df.drop_duplicates(subset=["text"], keep="first", inplace=True)
after_dedup = len(full_df)
print(f"\nRows before dedup: {before_dedup}")
print(f"Rows after dedup:  {after_dedup}")
print(f"Duplicates removed: {before_dedup - after_dedup}")

# ─────────────────────────────────────────────
# 5. Label Distribution
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Label Distribution")
print("=" * 60)

label_counts = full_df["label"].value_counts()
label_pcts   = full_df["label"].value_counts(normalize=True) * 100
label_map    = {1: "malicious", 0: "benign"}

print("\n--- Overall Label Distribution ---")
for label_val in sorted(label_counts.index):
    print(f"  {label_map.get(label_val, label_val)} ({label_val}): "
          f"{label_counts[label_val]} rows ({label_pcts[label_val]:.1f}%)")

print("\n--- Label Distribution by Split ---")
split_label = full_df.groupby(["split", "label"]).size().unstack(fill_value=0)
print(split_label.to_string())

# Plot label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall
colors = ["#2ecc71", "#e74c3c"]
label_counts.plot(kind="bar", ax=axes[0], color=colors)
axes[0].set_title("Overall Label Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Label (0=benign, 1=malicious)")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["Benign (0)", "Malicious (1)"], rotation=0)
for i, (val, count) in enumerate(label_counts.items()):
    axes[0].text(i, count + 20, f"{count}\n({label_pcts[val]:.1f}%)",
                 ha="center", fontweight="bold")

# By split
split_label.plot(kind="bar", ax=axes[1], color=colors)
axes[1].set_title("Label Distribution by Split", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Split")
axes[1].set_ylabel("Count")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(["Benign (0)", "Malicious (1)"])

plt.tight_layout()
plt.savefig("label_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: label_distribution.png")

# ─────────────────────────────────────────────
# 6. Category Distribution
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Category Distribution")
print("=" * 60)

cat_counts = full_df["category"].value_counts()
print("\n--- Category Counts ---")
for cat, count in cat_counts.items():
    pct = count / len(full_df) * 100
    print(f"  {cat:30s} : {count:5d} ({pct:5.1f}%)")

print(f"\nTotal unique categories: {full_df['category'].nunique()}")

# Plot category distribution
fig, ax = plt.subplots(figsize=(14, 7))
cat_counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(cat_counts)))
ax.set_title("Category Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Count")
ax.set_ylabel("Category")
ax.invert_yaxis()
for i, (count) in enumerate(cat_counts.values):
    ax.text(count + 5, i, str(count), va="center")
plt.tight_layout()
plt.savefig("category_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: category_distribution.png")

# ─────────────────────────────────────────────
# 7. Severity Distribution
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Severity Distribution")
print("=" * 60)

sev_counts = full_df["severity"].value_counts()
print("\n--- Severity Counts ---")
for sev, count in sev_counts.items():
    pct = count / len(full_df) * 100
    print(f"  {sev:10s} : {count:5d} ({pct:5.1f}%)")

# Severity by label
sev_by_label = full_df.groupby(["severity", "label"]).size().unstack(fill_value=0)
print("\n--- Severity by Label ---")
print(sev_by_label.to_string())

# Plot severity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

severity_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c", "critical": "#8e44ad"}
sev_order = ["low", "medium", "high", "critical"]
sev_ordered = sev_counts.reindex([s for s in sev_order if s in sev_counts.index])

sev_ordered.plot(kind="bar", ax=axes[0],
                 color=[severity_colors.get(s, "#95a5a6") for s in sev_ordered.index])
axes[0].set_title("Severity Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Severity")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Severity × Label heatmap
sev_by_label_reindexed = sev_by_label.reindex([s for s in sev_order if s in sev_by_label.index])
sns.heatmap(sev_by_label_reindexed, annot=True, fmt="d", cmap="YlOrRd", ax=axes[1])
axes[1].set_title("Severity × Label Heatmap", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Severity")
axes[1].set_xlabel("Label (0=benign, 1=malicious)")

plt.tight_layout()
plt.savefig("severity_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: severity_distribution.png")

# ─────────────────────────────────────────────
# 8. Text Length Analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Text Length Analysis")
print("=" * 60)

full_df["text_length"] = full_df["text"].str.len()
full_df["word_count"]  = full_df["text"].str.split().str.len()

print("\n--- Text Length Stats (characters) ---")
print(full_df.groupby("label")["text_length"].describe().to_string())

print("\n--- Word Count Stats ---")
print(full_df.groupby("label")["word_count"].describe().to_string())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

full_df[full_df["label"] == 0]["text_length"].hist(ax=axes[0], bins=50,
    alpha=0.7, color="#2ecc71", label="Benign")
full_df[full_df["label"] == 1]["text_length"].hist(ax=axes[0], bins=50,
    alpha=0.7, color="#e74c3c", label="Malicious")
axes[0].set_title("Text Length Distribution (chars)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Character Count")
axes[0].set_ylabel("Frequency")
axes[0].legend()

full_df[full_df["label"] == 0]["word_count"].hist(ax=axes[1], bins=50,
    alpha=0.7, color="#2ecc71", label="Benign")
full_df[full_df["label"] == 1]["word_count"].hist(ax=axes[1], bins=50,
    alpha=0.7, color="#e74c3c", label="Malicious")
axes[1].set_title("Word Count Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("text_length_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: text_length_distribution.png")

# ─────────────────────────────────────────────
# 9. Category × Label Cross-Tab
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Category × Label Cross-Tabulation")
print("=" * 60)

cat_label = pd.crosstab(full_df["category"], full_df["label"], margins=True)
cat_label.columns = ["benign (0)", "malicious (1)", "Total"]
cat_label = cat_label.sort_values("Total", ascending=False)
print("\n" + cat_label.to_string())

# ─────────────────────────────────────────────
# 10. Save cleaned dataset
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Saving Cleaned Dataset")
print("=" * 60)

# Remove helper columns before saving
save_df = full_df.drop(columns=["text_length", "word_count"], errors="ignore")

# Save combined cleaned dataset
output_path = "cleaned_neuralchemy_dataset.jsonl"
save_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
print(f"\nSaved cleaned dataset to: {output_path}")
print(f"Final shape: {save_df.shape}")
print(f"Final columns: {list(save_df.columns)}")

# Also save split-wise for later use
for split_name in ["train", "validation", "test"]:
    split_data = save_df[save_df["split"] == split_name]
    split_path = f"cleaned_neuralchemy_{split_name}.jsonl"
    split_data.to_json(split_path, orient="records", lines=True, force_ascii=False)
    print(f"  {split_name:12s} → {split_path} ({len(split_data)} rows)")

# ─────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
Dataset: neuralchemy/Prompt-injection-dataset (core split)
Total rows after cleaning: {len(save_df)}
Columns: {list(save_df.columns)}
Labels: benign={len(save_df[save_df['label']==0])}, malicious={len(save_df[save_df['label']==1])}
Categories: {save_df['category'].nunique()} unique
Splits: train={len(save_df[save_df['split']=='train'])}, val={len(save_df[save_df['split']=='validation'])}, test={len(save_df[save_df['split']=='test'])}

Files saved:
  - cleaned_neuralchemy_dataset.jsonl       (all splits combined)
  - cleaned_neuralchemy_train.jsonl          (train only)
  - cleaned_neuralchemy_validation.jsonl     (validation only)
  - cleaned_neuralchemy_test.jsonl           (test only)
  - label_distribution.png
  - category_distribution.png
  - severity_distribution.png
  - text_length_distribution.png
""")

print("✅ Done! Ready for augmentation pipeline.")
