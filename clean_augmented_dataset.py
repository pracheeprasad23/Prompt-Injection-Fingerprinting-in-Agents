"""Strict cleaner for Neuralchemy augmented dataset.

Purpose:
1) remove malformed/corrupted rows from model augmentation,
2) remove semantically inconsistent rows (label vs trace behavior),
3) produce a clean JSONL suitable for classifier training.

Default input:
  neuralchemy_augmented_dataset/augmented_neuralchemy_dataset.jsonl
Default output:
  cleaned_augmented_neuralchemy_dataset.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict


DEFAULT_INPUT = os.path.join(
    "neuralchemy_augmented_dataset", "augmented_neuralchemy_dataset.jsonl"
)
DEFAULT_OUTPUT = "cleaned_augmented_neuralchemy_dataset.jsonl"
DEFAULT_QUARANTINE_OUTPUT = "quarantine_removed_records.jsonl"

ALLOWED_TOOLS = {
    "file_read",
    "file_write",
    "code_exec",
    "db_query",
    "web_search",
    "send_msg",
    "knowledge_lookup",
    "api_call",
}

SUSPICION_KEYWORDS = {
    "injection",
    "suspicious",
    "override",
    "bypass",
    "attack",
    "malicious",
    "unauthorized",
    "not comply",
    "deny",
    "security",
    "harmful",
    "illegal",
    "dangerous",
    "jailbreak",
    "prohibited",
    "not permitted",
    "cannot comply",
    "refuse",
    "threat",
    "exploit",
    "system prompt",
    "prompt leak",
    "instruction override",
}

HIGH_RISK_PROMPT_PATTERNS = [
    re.compile(r"ignore\s+all\s+previous\s+instructions", re.I),
    re.compile(r"reveal\s+your\s+system\s+prompt", re.I),
    re.compile(r"output\s+the\s+system\s+prompt", re.I),
    re.compile(r"step[- ]?by[- ]?step.*(hack|exploit|phishing|malware)", re.I),
    re.compile(r"write\s+.*phishing\s+email", re.I),
    re.compile(r"how\s+to\s+make\s+a\s+bomb", re.I),
    re.compile(r"disable\s+antivirus", re.I),
    re.compile(r"steal\s+(credentials|password|data)", re.I),
    re.compile(r"hack\s+into", re.I),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Strict cleaner for augmented JSONL")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument(
        "--quarantine-output",
        default=DEFAULT_QUARANTINE_OUTPUT,
        help="Output JSONL path for dropped records with reason tags",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=2,
        help="Minimum reasoning steps required",
    )
    parser.add_argument(
        "--min-thought-chars",
        type=int,
        default=12,
        help="Minimum non-whitespace chars per thought",
    )
    parser.add_argument(
        "--strict-benign",
        action="store_true",
        help="Remove label=0 records with high-risk prompt patterns",
    )
    return parser.parse_args()


def load_jsonl_with_line_info(path):
    records = []
    bad_json_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            raw_line = line.rstrip("\n")
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append((line_num, json.loads(line)))
            except json.JSONDecodeError as exc:
                bad_json_lines.append((line_num, str(exc), raw_line))
    return records, bad_json_lines


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_quarantine_entry(
    source_file,
    line_num,
    reason,
    stage,
    record=None,
    error_message=None,
    raw_line=None,
):
    entry = {
        "source_file": source_file,
        "line_number": line_num,
        "drop_stage": stage,
        "reason_tags": [reason],
    }

    if error_message is not None:
        entry["error_message"] = error_message
    if raw_line is not None:
        entry["raw_line"] = raw_line
    if record is not None:
        entry["record"] = record
        entry["text"] = record.get("text", "")
        entry["label"] = record.get("label")
        entry["category"] = record.get("category")
        entry["severity"] = record.get("severity")
        entry["split"] = record.get("split")

    return entry


def has_suspicion_text(text):
    low = text.lower()
    return any(key in low for key in SUSPICION_KEYWORDS)


def is_high_risk_benign_prompt(prompt):
    return any(pattern.search(prompt) for pattern in HIGH_RISK_PROMPT_PATTERNS)


def first_non_string_index(values):
    for idx, value in enumerate(values):
        if not isinstance(value, str):
            return idx
    return -1


def validate_schema(rec):
    required = [
        "text",
        "label",
        "category",
        "severity",
        "split",
        "reasoning_trace",
        "tool_calls",
        "agent_final_action",
        "injection_detected_at_step",
        "raw_agent_trace",
    ]
    for key in required:
        if key not in rec:
            return False, f"missing_field:{key}"

    if not isinstance(rec["text"], str) or not rec["text"].strip():
        return False, "invalid_text"
    if rec["label"] not in (0, 1):
        return False, "invalid_label"
    if not isinstance(rec["category"], str) or not rec["category"].strip():
        return False, "invalid_category"
    if not isinstance(rec["split"], str) or not rec["split"].strip():
        return False, "invalid_split"
    if not isinstance(rec["agent_final_action"], str):
        return False, "invalid_agent_final_action"
    if not isinstance(rec["injection_detected_at_step"], int):
        return False, "invalid_injection_detected_at_step_type"
    if not isinstance(rec["raw_agent_trace"], str):
        return False, "invalid_raw_agent_trace"

    if not isinstance(rec["reasoning_trace"], list):
        return False, "invalid_reasoning_trace_type"
    non_str_idx = first_non_string_index(rec["reasoning_trace"])
    if non_str_idx != -1:
        return False, f"invalid_reasoning_trace_item:{non_str_idx}"

    if not isinstance(rec["tool_calls"], list):
        return False, "invalid_tool_calls_type"
    for idx, tool_call in enumerate(rec["tool_calls"]):
        if not isinstance(tool_call, dict):
            return False, f"invalid_tool_call_item:{idx}"
        for field in ("tool", "input", "output"):
            if field not in tool_call:
                return False, f"invalid_tool_call_missing:{idx}:{field}"
            if not isinstance(tool_call[field], str):
                return False, f"invalid_tool_call_type:{idx}:{field}"
        if tool_call["tool"] not in ALLOWED_TOOLS:
            return False, f"invalid_tool_name:{idx}:{tool_call['tool']}"

    return True, ""


def validate_quality(rec, min_steps, min_thought_chars, strict_benign=False):
    text = rec["text"].strip()
    label = rec["label"]
    trace = rec["reasoning_trace"]
    final_action = rec["agent_final_action"].strip()
    detected_at = rec["injection_detected_at_step"]
    raw = rec["raw_agent_trace"]

    if len(trace) == 0:
        return False, "empty_trace"
    if len(trace) < min_steps:
        return False, f"too_short_trace:{len(trace)}"

    # Detect parser fallback artifacts from augmentation script.
    if raw.startswith("[JSON_PARSE_FAILED]"):
        return False, "json_parse_failed_marker"

    cleaned_steps = [step.strip() for step in trace]
    if any(len(step) < min_thought_chars for step in cleaned_steps):
        return False, "short_or_low_info_thought"

    if detected_at < -1:
        return False, "invalid_detection_step_negative"
    if detected_at > len(trace):
        return False, "invalid_detection_step_out_of_range"

    joined_trace = " ".join(cleaned_steps)
    suspicion_in_trace_or_action = has_suspicion_text(joined_trace + " " + final_action)

    # Core label-trace consistency rules.
    if label == 1 and detected_at == -1 and not suspicion_in_trace_or_action:
        return False, "malicious_without_detection_signal"

    if label == 0 and detected_at != -1:
        return False, "benign_with_detection_step"

    if strict_benign and label == 0 and is_high_risk_benign_prompt(text):
        return False, "label_noise_high_risk_prompt"

    return True, ""


def summarize_labels(records):
    label_counter = Counter(rec.get("label", -1) for rec in records)
    return label_counter.get(0, 0), label_counter.get(1, 0)


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading: {args.input}")
    raw_records, bad_json_lines = load_jsonl_with_line_info(args.input)
    total_input_rows = len(raw_records)
    print(f"  Parsed rows: {total_input_rows}")
    print(f"  JSON decode failures while loading: {len(bad_json_lines)}")

    kept_records = []
    quarantine_records = []
    removed_reasons = Counter()
    removed_examples = defaultdict(list)
    all_records_for_stats = []

    # Capture line-level JSON parse failures in quarantine output.
    for line_num, msg, raw_line in bad_json_lines:
        quarantine_records.append(
            build_quarantine_entry(
                source_file=args.input,
                line_num=line_num,
                reason="file_json_decode_failure",
                stage="load",
                error_message=msg,
                raw_line=raw_line,
            )
        )

    for line_num, rec in raw_records:
        all_records_for_stats.append(rec)

        valid_schema, schema_reason = validate_schema(rec)
        if not valid_schema:
            removed_reasons[schema_reason] += 1
            quarantine_records.append(
                build_quarantine_entry(
                    source_file=args.input,
                    line_num=line_num,
                    reason=schema_reason,
                    stage="schema",
                    record=rec,
                )
            )
            if len(removed_examples[schema_reason]) < 3:
                removed_examples[schema_reason].append((line_num, rec.get("text", "")[:120]))
            continue

        valid_quality, quality_reason = validate_quality(
            rec,
            min_steps=args.min_steps,
            min_thought_chars=args.min_thought_chars,
            strict_benign=args.strict_benign,
        )
        if not valid_quality:
            removed_reasons[quality_reason] += 1
            quarantine_records.append(
                build_quarantine_entry(
                    source_file=args.input,
                    line_num=line_num,
                    reason=quality_reason,
                    stage="quality",
                    record=rec,
                )
            )
            if len(removed_examples[quality_reason]) < 3:
                removed_examples[quality_reason].append((line_num, rec.get("text", "")[:120]))
            continue

        kept_records.append(rec)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(kept_records, args.output)
    os.makedirs(os.path.dirname(args.quarantine_output) or ".", exist_ok=True)
    save_jsonl(quarantine_records, args.quarantine_output)

    before_benign, before_malicious = summarize_labels(all_records_for_stats)
    after_benign, after_malicious = summarize_labels(kept_records)

    total_removed = sum(removed_reasons.values()) + len(bad_json_lines)
    kept_count = len(kept_records)
    removed_pct = (total_removed / total_input_rows * 100.0) if total_input_rows else 0.0

    print("\n" + "=" * 72)
    print("STRICT CLEANING REPORT")
    print("=" * 72)

    print("\n[BEFORE]")
    print(f"  Total rows: {total_input_rows}")
    print(f"  Benign(0): {before_benign}")
    print(f"  Malicious(1): {before_malicious}")

    print("\n[REMOVED]")
    if bad_json_lines:
        print(f"  file_json_decode_failure: {len(bad_json_lines)}")
    for reason, count in removed_reasons.most_common():
        print(f"  {reason}: {count}")
    print(f"  Total removed: {total_removed} ({removed_pct:.2f}%)")

    print("\n[AFTER]")
    print(f"  Total rows: {kept_count}")
    print(f"  Benign(0): {after_benign}")
    print(f"  Malicious(1): {after_malicious}")
    if kept_count:
        print(f"  Balance: {after_benign / kept_count * 100:.1f}% benign / {after_malicious / kept_count * 100:.1f}% malicious")

    print(f"\nOutput saved to: {args.output}")
    print(f"Quarantine saved to: {args.quarantine_output}")

    if bad_json_lines:
        print("\nExamples [file_json_decode_failure]:")
        for line_num, msg, _ in bad_json_lines[:3]:
            print(f"  - line {line_num}: {msg}")

    for reason, examples in removed_examples.items():
        if examples:
            print(f"\nExamples [{reason}]:")
            for line_num, snippet in examples:
                print(f"  - line {line_num}: {snippet}")


if __name__ == "__main__":
    main()
