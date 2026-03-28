# Project 2 Paper — File Guide (Concise)

This README summarizes all project files except those explicitly ignored by `.gitignore` patterns (for example `Prompt_INJECTION_And_Benign_DATASET/`, `augment_dataset.py`, caches, IDE, venv).

## Core Project Philosophy & Collaboration Points

**Overall Goal**: Detect prompt injections in LLM inputs by training a classifier on augmented reasoning traces that show agent defensive behavior.

**Key Design Principles**:
1. **Augmentation as Signal**: Rather than collecting manual injection labels, we generate realistic LLM-agent execution traces using Qwen2.5-7B-Instruct. Malicious prompts trigger defensive tool calls (e.g., logging to security log) and explicit suspicion markers in reasoning.
2. **Strict Split Integrity**: Every record carries a `split` field (train/validation/test). Training/evaluation code uses this field to prevent leakage. No random train-test splits downstream.
3. **Composable Input Representation**: Classifiers ingest `[PROMPT] [THOUGHTS] [TOOLS] [ACTION] [STEP]` — a concatenated view of original text + augmented traces. Ablations systematically remove each component to measure contribution.
4. **Quality Gatekeeping**: Aggressive cleaning (schema validation, trace quality, label-trace consistency checks) ensures training data is high-confidence; rejected records are quarantined with audit reasons for inspection.
5. **Reproducible Evaluation**: Phase-4 evaluation (baselines, ablations, latency) produces a single JSON artifact (`results_phase4.json`) as source of truth; plots are regenerated from this JSON, not re-trained.

**Key files to understand first**:
- **Data flow**: `neuralchemy_original_dataset/dataset_download_and_eda.py` → `augment_neuralchemy_dataset.py` → `clean_augmented_dataset.py` → `classifier_model/train_classifier_kaggle.py`.
- **Model training config**: `classifier_model/train_classifier_kaggle.py` — hyperparameters, architecture, loss function, split handling.
- **Component contribution**: `classifier_model/evaluate_baselines_kaggle.py` — ablation study reveals which augmented signals matter most.
- **Generated data issues**: `quarantine_removed_records.jsonl` — inspect rejection patterns to debug augmentation pipeline.
- **Paper artifact**: `classifier_model/phase4_results_archive/results_phase4.json` — all metrics used in paper claims.

## End-to-End Flow
1. Download and clean original dataset: `neuralchemy_original_dataset/dataset_download_and_eda.py`.
2. Augment with ReAct traces/tools/actions: `augment_neuralchemy_dataset.py` (or notebook variant).
3. Strict-clean augmented data: `clean_augmented_dataset.py`.
4. Run EDA on cleaned augmented data: `eda_augmented_dataset.py`.
5. Train classifier on split-aware data: `classifier_model/train_classifier_kaggle.py`.
6. Evaluate baselines/ablations/latency and export plots/JSON: `classifier_model/evaluate_baselines_kaggle.py`.
7. Optionally re-plot clean figures from JSON: `classifier_model/phase4_results_archive/results_from_json.ipynb`.

## Important Notes
- Split integrity: training/evaluation scripts use `split` field (`train/validation/test`) for strict separation.
- Input representation for classifier uses combined fields: `[PROMPT] [THOUGHTS] [TOOLS] [ACTION] [STEP]`.
- `results_phase4.json` is the central paper-results artifact (metrics source of truth).

---

## File-by-File Summary

### Root-level files

- `augment-neuralchemy-dataset.ipynb`
  - **Purpose**: Interactive Kaggle notebook wrapper for the augmentation pipeline.
  - **Workflow**:
    1. Install dependencies (bitsandbytes, accelerate, datasets, transformers, etc.).
    2. Upload or reference input dataset (HF dataset or JSONL).
    3. Call `augment_neuralchemy_dataset.py` script inline with optional flags (`--input`, `--output`, `--max`, `--start`).
    4. Monitor checkpoints and progress in real-time (every 50 records).
    5. Download augmented JSONL on completion.
  - **Typical Kaggle Usage**:
    - Dataset: `/kaggle/input/` contains cleaned base dataset.
    - Script: Cell runs augmentation with GPU acceleration.
    - Output: `/kaggle/working/` produces augmented JSONL + checkpoint files.
  - **Design Focus**: Enables collaborative augmentation on Kaggle (11-hour timeout-friendly, checkpointed, resumable).

- `augment_neuralchemy_dataset.py`
  - **Purpose**: LLM-based trace augmentation using Qwen2.5-7B-Instruct (4-bit quantization).
  - **Generation**: Few-shot prompted ReAct traces with 8 enterprise tools. Output: JSON with steps, final_action, injection_detected_at_step.
  - **Quality Assurance**: Validates 2–8 steps, checks tool names, removes duplicate thoughts, retries up to 3x.
  - **Checkpointing**: Saves every 10 records, 11-hour timeout recovery, resume support.
  - **Output**: `reasoning_trace`, `tool_calls`, `agent_final_action`, `injection_detected_at_step`, `raw_agent_trace`.

- `clean_augmented_dataset.py`
  - **Purpose**: Strict quality validation & cleaning for augmented records.
  - **Validation Stages**: 
    - Schema: required fields, types (text string, label ∈ {0,1}, non-empty category/split).
    - Trace Quality: 2–8 steps, ≥12 chars/thought, valid tool names, no duplicate thoughts.
    - Label-Trace Consistency: malicious records have suspicion keywords + `injection_detected_at_step` ≥ 1; benign records lack high-risk patterns.
  - **Output & Quarantine**: Cleaned JSONL + `quarantine_removed_records.jsonl` (with rejection reasons for debugging).

- `cleaned_augmented_neuralchemy_dataset.jsonl`
  - **Purpose**: Master dataset for training/evaluation (all records pass quality checks).
  - **Fields**: Original (text, label, category, severity, split) + Augmented (reasoning_trace, tool_calls, agent_final_action, injection_detected_at_step, raw_agent_trace).

- `eda_augmented_dataset.py`
  - **Purpose**: EDA & visualization for cleaned augmented dataset.
  - **Outputs**: Console summary + 11 PNG plots (label/category/severity/split/steps/thoughts/tools/sequences/injection-detection/actions).

- `quarantine_removed_records.jsonl`
  - **Purpose**: Audit log of rejected records (enables debugging & pattern analysis).
  - **Fields**: source_file, line_number, drop_stage, reason_tags, error_message, record snapshot, extracted fields.

### `classifier_model/`

- `classifier_model/cleaned_augmented_neuralchemy_dataset.jsonl`
  - Purpose: local copy of cleaned dataset for classifier workflows.
  - Logic: duplicate/synced dataset copy for Kaggle/model folder convenience.
  - Input/Output: consumed by training/eval scripts.

- `classifier_model/train_classifier_kaggle.py`
  - **Purpose**: Single-GPU trainer for binary prompt-injection classification.
  - **Model**: `microsoft/deberta-v3-small` (~70M params), binary output, float32 precision.
  - **Hyperparameters**: epochs=5, batch_size=8, lr=8e-6, dropout=0.3, label_smoothing=0.08, max_len=256, grad_accum=2.
  - **Loss**: Weighted CrossEntropyLoss (class-weighted) + label smoothing.
  - **Optimization**: AdamW (weight_decay=0.01), linear warmup (10%) → decay, gradient clipping (norm=1.0).
  - **Early Stopping**: patience=2 on validation F1(malicious).
  - **Input Format**: `[PROMPT] [THOUGHTS] [TOOLS] [ACTION] [STEP]` from augmented data.
  - **Outputs**: `best_binary_injection/` checkpoint + metrics JSON/TXT.
  - **Design Focus**: Balanced class weights + label smoothing prevent overfitting; strict split integrity (train/val/test).

- `classifier_model/train_classifier_kaggle_noteboook.ipynb`
  - Purpose: Kaggle notebook orchestration for training.
  - Logic: install deps, copy assets from `/kaggle/input`, run trainer, zip artifacts.
  - Input: dataset + trainer script.
  - Output: trained model artifacts ZIP.

- `classifier_model/evaluate_baselines_kaggle.py`
  - **Purpose**: Phase-4 evaluation (baselines, ablations, latency, paper metrics).
  - **Baselines**: TF-IDF+LR (ML baseline), prompt-only DeBERTa (DL baseline), full model (with traces).
  - **Ablations**: Full model, w/o reasoning trace, w/o tool calls, w/o injection step.
  - **Metrics**: F1 (weighted/macro/binary), AUC, confusion matrices, latency (ms/sample, throughput).
  - **Outputs**: `results_phase4.json` (source of truth) + PNG plots (confusion/F1/ablation/ROC/latency).
  - **Design Focus**: Ablations justify component contributions; consolidated JSON enables figure regeneration.

- `classifier_model/evaluate_baselines_kaggle_notebook.ipynb`
  - Purpose: Kaggle execution notebook for phase-4 evaluation.
  - Logic: copies script/data/model from `/kaggle/input`, handles folder/zip model layouts, runs evaluation, zips outputs.
  - Input: dataset, eval script, model artifacts.
  - Output: phase-4 result files in Kaggle working/output.

- `classifier_model/phase4_results_archive.zip`
  - Purpose: packaged phase-4 outputs.
  - Logic: zipped result artifact.
  - Input: generated phase-4 files.
  - Output: distributable archive.

### `classifier_model/phase4_results_archive/`

- `classifier_model/phase4_results_archive/results_phase4.json`
  - **Purpose**: Single source of truth for all paper metrics (baselines, ablations, latency, metadata).
  - **Contents**: baseline F1/AUC scores, ablation F1 scores, latency stats, model/dataset metadata.
  - **Usage**: Consumed by `results_from_json.ipynb` to regenerate publication plots without re-training.

- `classifier_model/phase4_results_archive/results_from_json.ipynb`
  - **Purpose**: Regenerate clean plots from `results_phase4.json` for paper revisions.
  - **Plots Generated**: baseline F1 comparison, ablation F1, ROC curves, confusion matrices, AUC, latency benchmark.
  - **Design Focus**: Decouples visualization from training; enables rapid figure iteration.

- `classifier_model/phase4_results_archive/` — Phase-4 plots:
  - Original: confusion matrices, F1/ablation/baseline comparisons, ROC curves, latency benchmark.
  - Regenerated from JSON in `graphs_from_json/`: same plots + AUC chart for publication.

### `eda_augmented_plots/` (generated by augmented EDA script)

- `eda_augmented_plots/` — Generated visualizations:
  - Label distribution (benign/malicious overall and by split).
  - Category, severity, and split distributions.
  - Trace statistics (step counts, thought lengths, tool usage patterns).
  - Injection detection step distribution and final action lengths.
  - 11 PNG files total documenting dataset characteristics.

### `neuralchemy_augmented_dataset/`

- `neuralchemy_augmented_dataset/augmented_neuralchemy_dataset.jsonl`
  - Purpose: primary augmented dataset output.
  - Input: base records + generated traces.
  - Output: input to strict cleaner.

- `neuralchemy_augmented_dataset/augmented_neuralchemy_dataset (1-3).jsonl`
  - Purpose: snapshot exports from iterative augmentation runs (backups/comparison).

### `neuralchemy_original_dataset/`

- `neuralchemy_original_dataset/dataset_download_and_eda.py`
  - **Purpose**: Foundation script — downloads & cleans original dataset from HF, performs baseline EDA.
  - **Processing**: Downloads HF core split, removes metadata columns (`augmented`, `group_id`, `source`, `tags`), deduplicates, adds explicit `split` field (train/validation/test), validates split integrity (no leakage).
  - **Outputs**: Cleaned JSONLs (combined + split-specific), PNG plots (label/category/severity/text-length distributions), HF metadata.
  - **Design Focus**: Clean, well-understood baseline before augmentation; split tags enable strict training/eval separation downstream.

- `neuralchemy_original_dataset/cleaned_neuralchemy_*.jsonl`
  - Purpose: cleaned dataset exports: combined (`_dataset.jsonl`), split-specific (`_train/validation/test.jsonl`), and alternates (`_v2.jsonl`).

- `neuralchemy_original_dataset/` — Generated EDA plots:
  - Label, category, severity, and text-length distributions (4 PNG files).
  - `__huggingface_repos__.json`: HF dataset fetch metadata.

- `neuralchemy_original_dataset/__results___files/` — Archive of auxiliary result images from earlier notebook runs.


---

## Technical Design Decisions & Trade-Offs (Collaborator Notes)

### Augmentation (`augment_neuralchemy_dataset.py`)
- **Model Choice: Qwen2.5-7B-Instruct (4-bit)**
  - Why: Strong reasoning capabilities, open-source, efficient at 4-bit quantization.
  - Trade-off: Generates synthetic traces (not ground truth). Validation logic catches common failures (duplicate thoughts, invalid tools), but hallucinations can slip through.
- **Few-Shot Prompt with Chat Template**
  - Why: Few-shot examples (benign + malicious) guide consistent output format; chat template prevents model confusion.
  - Trade-off: Only 2 examples. More examples = stricter behavior but longer context.
- **8-Tool Simulation**
  - Why: Mimics realistic enterprise agent tooling. When agents use defensive tools (file_write to security log), it signals injection detection.
  - Trade-off: Synthetic tool results are simulated, not real. Doesn't capture all real attack scenarios.

### Cleaning (`clean_augmented_dataset.py`)
- **Three-Stage Validation** (schema → trace quality → label-trace consistency)
  - Why: Catches most errors early (malformed JSON), then trace issues (2–8 steps), then semantic mismatches (benign with high-risk patterns).
  - Trade-off: May over-reject edge cases (e.g., lengthy complex reasoning is truncated if >8 steps). Can adjust `--min-steps`, `--min-thought-chars` params.
- **Quarantine Instead of Drop**
  - Why: Enables post-hoc analysis of rejects; helps debug augmentation pipeline.
  - Trade-off: Storage overhead. Quarantine file can be large if many records fail.

### Training (`classifier_model/train_classifier_kaggle.py`)
- **DeBERTa v3-Small (70M params)**
  - Why: Lightweight, pre-trained on large corpus, strong on token-level attention. Balances accuracy vs. inference latency.
  - Trade-off: Smaller than base/large variants; may leave accuracy on the table. Larger models are slower.
- **Weighted Cross-Entropy Loss + Label Smoothing (0.08)**
  - Why: Weighted CE handles class imbalance (more benign examples in dataset); label smoothing (0.08) prevents overconfidence and reduces overfitting.
  - Trade-off: Weakens model's ability to be fully confident on clean benign examples. Too much smoothing → underfitting; too little → overfitting.
- **Input Template: `[PROMPT] [THOUGHTS] [TOOLS] [ACTION] [STEP]`**
  - Why: Concatenation is simple, fast, avoids need for multi-input architectures. Gives model all signals in one sequence.
  - Trade-off: No explicit multi-head attention over individual signals; ablations test removal of entire blocks (not fine-grained importance).

### Evaluation (`classifier_model/evaluate_baselines_kaggle.py`)
- **Ablation**: Removes one signal at a time to measure component contribution. Doesn't capture interaction effects but provides straightforward causal inference.
- **Three Baselines**: TF-IDF+LR (statistical), prompt-only DeBERTa (deep learning only), full model (with traces). Trade-off: baselines are generic; specialized models could be stronger.
- **JSON Output** (`results_phase4.json`): Single source of truth; enables plot regeneration without re-training. Risk: hand-edits diverge from code.

### Data Organization & Reproducibility
- **Split Field**: Prevents leakage by explicit `split == "train"/"validation"/"test"` checks. Must keep in sync across copies.
- **Separate Dataset Copies**: `classifier_model/cleaned_augmented_neuralchemy_dataset.jsonl` duplicates root for Kaggle workflow convenience. Risk: synchronization drift.
- **Checkpointing** (save every 10, report every 50): Enables Kaggle 11-hour timeout recovery. Trade-off: overhead for small datasets.
- **Deterministic Seeds** (seed=42): Reproducible but limits ensemble diversity. Single run has no error bars.
