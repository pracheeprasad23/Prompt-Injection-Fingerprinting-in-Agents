"""
EDA on Cleaned Augmented Dataset
=================================
Generates comprehensive insights on the cleaned augmented neuralchemy dataset,
covering both original fields and augmented trace features.

Input:  cleaned_augmented_neuralchemy_dataset.jsonl
Output: Console report + visualization plots saved as PNGs
"""

import json
import os
import sys
from collections import Counter

# ─── Try importing plotting libraries ───
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_PLOTS = True
except ImportError:
    HAS_PLOTS = False
    print("[INFO] matplotlib not found — skipping plots, text report only.\n")

# ─── Configuration ───
INPUT_FILE = "cleaned_augmented_neuralchemy_dataset.jsonl"
PLOT_DIR   = "eda_augmented_plots"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_dist(counter, label="", top_n=None):
    items = counter.most_common(top_n)
    total = sum(counter.values())
    for k, v in items:
        pct = v / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {str(k):30s} : {v:5d}  ({pct:5.1f}%)  {bar}")


def save_bar_chart(counter, title, filename, xlabel="", ylabel="Count", top_n=None, color="#4A90D9"):
    if not HAS_PLOTS:
        return
    items = counter.most_common(top_n)
    labels = [str(k) for k, v in items]
    values = [v for k, v in items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    bars = ax.barh(labels[::-1], values[::-1], color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


def save_pie_chart(counter, title, filename):
    if not HAS_PLOTS:
        return
    labels = [str(k) for k in counter.keys()]
    values = list(counter.values())
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors[:len(labels)], textprops={"fontsize": 11}
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


def save_histogram(data, title, filename, xlabel="", bins=10, color="#4A90D9"):
    if not HAS_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


def save_split_label_chart(split_label_counts, title, filename):
    if not HAS_PLOTS:
        return

    ordered_splits = ["train", "validation", "test"]
    splits_present = [s for s in ordered_splits if s in split_label_counts]
    for split in sorted(split_label_counts.keys()):
        if split not in splits_present:
            splits_present.append(split)

    benign_counts = [split_label_counts[s].get(0, 0) for s in splits_present]
    malicious_counts = [split_label_counts[s].get(1, 0) for s in splits_present]

    x = list(range(len(splits_present)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - width / 2 for i in x], benign_counts, width=width, color="#2ECC71", label="benign (0)")
    b2 = ax.bar([i + width / 2 for i in x], malicious_counts, width=width, color="#E74C3C", label="malicious (1)")

    ax.set_xticks(x)
    ax.set_xticklabels(splits_present)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for bar in list(b1) + list(b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close()


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        sys.exit(1)

    if HAS_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)

    records = load_jsonl(INPUT_FILE)
    total = len(records)
    print(f"Loaded {total} records from {INPUT_FILE}")

    # ── Extract all fields ──
    labels = []
    categories = []
    severities = []
    splits = []
    step_counts = []
    tool_call_counts = []
    tool_names_all = []
    thought_lengths = []
    detected_steps = []
    final_action_lengths = []
    prompt_lengths = []
    tool_sequences = []

    for rec in records:
        labels.append(rec.get("label", -1))
        categories.append(rec.get("category", "unknown"))
        severities.append(rec.get("severity", "") or "none")
        splits.append(rec.get("split", "unknown"))
        prompt_lengths.append(len(rec.get("text", "")))

        trace = rec.get("reasoning_trace", [])
        tools = rec.get("tool_calls", [])
        final = rec.get("agent_final_action", "")
        det = rec.get("injection_detected_at_step", -1)

        step_counts.append(len(trace))
        tool_call_counts.append(len(tools))
        detected_steps.append(det)
        final_action_lengths.append(len(final))

        for t in trace:
            thought_lengths.append(len(t))

        tool_seq = []
        for tc in tools:
            name = tc.get("tool", "unknown")
            tool_names_all.append(name)
            tool_seq.append(name)
        tool_sequences.append(" -> ".join(tool_seq) if tool_seq else "no_tools")

    # ══════════════════════════════════════════
    # SECTION 1: Basic Dataset Overview
    # ══════════════════════════════════════════
    print_section("1. BASIC OVERVIEW")
    print(f"  Total records     : {total}")
    print(f"  Avg prompt length : {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")
    print(f"  Min/Max prompt    : {min(prompt_lengths)} / {max(prompt_lengths)} chars")

    # ══════════════════════════════════════════
    # SECTION 2: Label Distribution
    # ══════════════════════════════════════════
    print_section("2. LABEL DISTRIBUTION")
    label_counter = Counter(labels)
    label_named = Counter({"benign (0)": label_counter.get(0, 0), "malicious (1)": label_counter.get(1, 0)})
    print_dist(label_named)
    save_pie_chart(label_named, "Label Distribution", "01_label_distribution.png")

    print("\n  Label distribution by split:")
    split_label_counts = {}
    for split_name, label in zip(splits, labels):
        if split_name not in split_label_counts:
            split_label_counts[split_name] = Counter()
        split_label_counts[split_name][label] += 1

    ordered_splits = ["train", "validation", "test"]
    splits_present = [s for s in ordered_splits if s in split_label_counts]
    for split_name in sorted(split_label_counts.keys()):
        if split_name not in splits_present:
            splits_present.append(split_name)

    for split_name in splits_present:
        benign_count = split_label_counts[split_name].get(0, 0)
        malicious_count = split_label_counts[split_name].get(1, 0)
        split_total = benign_count + malicious_count
        benign_pct = (benign_count / split_total * 100) if split_total else 0
        malicious_pct = (malicious_count / split_total * 100) if split_total else 0
        print(
            f"    {split_name:10s} -> benign: {benign_count:4d} ({benign_pct:5.1f}%) | "
            f"malicious: {malicious_count:4d} ({malicious_pct:5.1f}%)"
        )

    save_split_label_chart(
        split_label_counts,
        "Label Distribution by Split",
        "01b_label_distribution_by_split.png",
    )

    # ══════════════════════════════════════════
    # SECTION 3: Category Distribution
    # ══════════════════════════════════════════
    print_section("3. CATEGORY DISTRIBUTION")
    cat_counter = Counter(categories)
    print_dist(cat_counter)
    save_bar_chart(cat_counter, "Category Distribution", "02_category_distribution.png")

    # ══════════════════════════════════════════
    # SECTION 4: Severity Distribution
    # ══════════════════════════════════════════
    print_section("4. SEVERITY DISTRIBUTION")
    sev_counter = Counter(severities)
    print_dist(sev_counter)
    save_bar_chart(sev_counter, "Severity Distribution", "03_severity_distribution.png", color="#E74C3C")

    # ══════════════════════════════════════════
    # SECTION 5: Split Distribution
    # ══════════════════════════════════════════
    print_section("5. SPLIT DISTRIBUTION")
    split_counter = Counter(splits)
    print_dist(split_counter)
    save_pie_chart(split_counter, "Split Distribution", "04_split_distribution.png")

    # ══════════════════════════════════════════
    # SECTION 6: Reasoning Trace Stats
    # ══════════════════════════════════════════
    print_section("6. REASONING TRACE ANALYSIS")
    step_counter = Counter(step_counts)
    avg_steps = sum(step_counts) / len(step_counts)
    avg_thought_len = sum(thought_lengths) / len(thought_lengths) if thought_lengths else 0
    print(f"  Avg steps/record    : {avg_steps:.1f}")
    print(f"  Min/Max steps       : {min(step_counts)} / {max(step_counts)}")
    print(f"  Avg thought length  : {avg_thought_len:.0f} chars")
    print(f"  Min/Max thought len : {min(thought_lengths)} / {max(thought_lengths)} chars")
    print(f"\n  Step count distribution:")
    print_dist(step_counter)
    save_bar_chart(step_counter, "Reasoning Steps per Record", "05_step_distribution.png", color="#2ECC71")
    save_histogram(thought_lengths, "Thought Length Distribution", "06_thought_lengths.png",
                   xlabel="Characters per thought", bins=20, color="#9B59B6")

    # ══════════════════════════════════════════
    # SECTION 7: Tool Call Analysis
    # ══════════════════════════════════════════
    print_section("7. TOOL CALL ANALYSIS")
    tool_count_counter = Counter(tool_call_counts)
    tool_name_counter = Counter(tool_names_all)
    avg_tools = sum(tool_call_counts) / len(tool_call_counts)
    print(f"  Total tool calls    : {len(tool_names_all)}")
    print(f"  Avg tools/record    : {avg_tools:.1f}")
    print(f"  Records with 0 tools: {tool_call_counts.count(0)}")
    print(f"\n  Tool calls per record:")
    print_dist(tool_count_counter)
    print(f"\n  Tool usage frequency:")
    print_dist(tool_name_counter)
    save_bar_chart(tool_name_counter, "Tool Usage Frequency", "07_tool_usage.png", color="#F39C12")
    save_bar_chart(tool_count_counter, "Tool Calls per Record", "08_tools_per_record.png", color="#1ABC9C")

    # ══════════════════════════════════════════
    # SECTION 8: Top Tool Sequences
    # ══════════════════════════════════════════
    print_section("8. TOP TOOL SEQUENCES")
    seq_counter = Counter(tool_sequences)
    print_dist(seq_counter, top_n=15)
    save_bar_chart(seq_counter, "Top 15 Tool Sequences", "09_tool_sequences.png", top_n=15, color="#E67E22")

    # ══════════════════════════════════════════
    # SECTION 9: Injection Detection
    # ══════════════════════════════════════════
    print_section("9. INJECTION DETECTION ANALYSIS")
    det_counter = Counter(detected_steps)
    not_detected = detected_steps.count(-1)
    detected = total - not_detected
    print(f"  Injection detected     : {detected} ({detected/total*100:.1f}%)")
    print(f"  Not detected (normal)  : {not_detected} ({not_detected/total*100:.1f}%)")
    print(f"\n  Detection step distribution:")
    print_dist(det_counter)
    save_bar_chart(det_counter, "Injection Detection Step", "10_injection_detection.png", color="#E74C3C")

    # Cross: label vs detection
    print(f"\n  Cross-tab: Label vs Detection")
    mal_detected = sum(1 for i in range(total) if labels[i] == 1 and detected_steps[i] >= 1)
    mal_not = sum(1 for i in range(total) if labels[i] == 1 and detected_steps[i] == -1)
    ben_detected = sum(1 for i in range(total) if labels[i] == 0 and detected_steps[i] >= 1)
    ben_not = sum(1 for i in range(total) if labels[i] == 0 and detected_steps[i] == -1)
    print(f"  Malicious + detected   : {mal_detected}")
    print(f"  Malicious + undetected : {mal_not}")
    print(f"  Benign + detected      : {ben_detected} (false positives)")
    print(f"  Benign + undetected    : {ben_not}")

    # ══════════════════════════════════════════
    # SECTION 10: Label vs Features (for classifier insight)
    # ══════════════════════════════════════════
    print_section("10. LABEL vs FEATURES (Classifier Insight)")

    # Avg steps by label
    benign_steps = [step_counts[i] for i in range(total) if labels[i] == 0]
    mal_steps = [step_counts[i] for i in range(total) if labels[i] == 1]
    print(f"  Avg steps (benign)    : {sum(benign_steps)/len(benign_steps):.1f}")
    print(f"  Avg steps (malicious) : {sum(mal_steps)/len(mal_steps):.1f}")

    # Avg tool calls by label
    benign_tools = [tool_call_counts[i] for i in range(total) if labels[i] == 0]
    mal_tools = [tool_call_counts[i] for i in range(total) if labels[i] == 1]
    print(f"  Avg tools (benign)    : {sum(benign_tools)/len(benign_tools):.1f}")
    print(f"  Avg tools (malicious) : {sum(mal_tools)/len(mal_tools):.1f}")

    # Avg prompt length by label
    benign_plen = [prompt_lengths[i] for i in range(total) if labels[i] == 0]
    mal_plen = [prompt_lengths[i] for i in range(total) if labels[i] == 1]
    print(f"  Avg prompt len (benign)   : {sum(benign_plen)/len(benign_plen):.0f} chars")
    print(f"  Avg prompt len (malicious): {sum(mal_plen)/len(mal_plen):.0f} chars")

    # Tool preference by label
    benign_tool_names = []
    mal_tool_names = []
    for i, rec in enumerate(records):
        for tc in rec.get("tool_calls", []):
            if labels[i] == 0:
                benign_tool_names.append(tc.get("tool", ""))
            else:
                mal_tool_names.append(tc.get("tool", ""))

    print(f"\n  Tool preference (benign):")
    print_dist(Counter(benign_tool_names))
    print(f"\n  Tool preference (malicious):")
    print_dist(Counter(mal_tool_names))

    # ══════════════════════════════════════════
    # SECTION 11: Final Action Length
    # ══════════════════════════════════════════
    print_section("11. FINAL ACTION LENGTH")
    avg_fa = sum(final_action_lengths) / len(final_action_lengths)
    print(f"  Avg length : {avg_fa:.0f} chars")
    print(f"  Min/Max    : {min(final_action_lengths)} / {max(final_action_lengths)} chars")
    save_histogram(final_action_lengths, "Final Action Length Distribution",
                   "11_final_action_length.png", xlabel="Characters", bins=20)

    # ══════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EDA COMPLETE")
    print(f"{'='*60}")
    if HAS_PLOTS:
        print(f"  Plots saved to: {PLOT_DIR}/")
        print(f"  Files: {', '.join(sorted(os.listdir(PLOT_DIR)))}")
    print(f"  Total records analyzed: {total}")


if __name__ == "__main__":
    main()
