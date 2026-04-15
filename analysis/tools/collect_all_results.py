from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


MODEL_RE = re.compile(r"^(?P<family>gemma2|llama3|qwen3)-(?P<size>\d+b)-(?P<rest>.+)$")
SEED_RE = re.compile(r"-seed(?P<seed>\d+)(?P<suffix>(?:-.*)?)$")
METHOD_RE = re.compile(
    r"^(?P<method>[a-zA-Z0-9]+?)(?:-(?P<rank_kind>[kr])(?P<rank_or_k>\d+))?(?:-(?P<variant>.+))?$"
)
ALPHA_RE = re.compile(r"(?:^|-)a(?P<alpha>\d+(?:\.\d+)?)(?:-|$)")
SEQ_RE = re.compile(r"(?:^|-)s(?P<seq_len>\d+)(?:-|$)")
PLR_RE = re.compile(r"plr(?P<plr>[\d.eE+-]+)")

METRIC_KEYS = [
    "Cl",
    "Fgt",
    "STEM_acc",
    "STEM_forgetting",
    "humanities_acc",
    "humanities_forgetting",
    "other (business, health, misc.)_acc",
    "other (business, health, misc.)_forgetting",
]

SCALED_METRIC_KEYS = {
    "STEM_acc",
    "STEM_forgetting",
    "humanities_acc",
    "humanities_forgetting",
    "other (business, health, misc.)_acc",
    "other (business, health, misc.)_forgetting",
}


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                data.append(obj)
    return data


def choose_metric_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    # The project writes summary metrics in the first two lines.
    merged: Dict[str, Any] = {}
    for row in rows[:2]:
        merged.update(row)
    return merged


def safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def scale_metric_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * 100.0
    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return value


def parse_run_name(run_dir: Path, output_root: Path) -> Dict[str, Any]:
    relative_name = run_dir.relative_to(output_root).as_posix() if run_dir.is_relative_to(output_root) else run_dir.name
    base_name = run_dir.name

    parsed: Dict[str, Any] = {
        "run_name": relative_name,
        "model_name": None,
        "method": None,
        "method_rank": None,
        "seed": None,
        "alpha": None,
        "seq_len": None,
        "plr": None,
        # extra_suffix and parse_status intentionally omitted per user request
    }

    model_match = MODEL_RE.match(base_name)
    if not model_match:
        return parsed

    model_family = model_match.group("family")
    model_size = model_match.group("size")
    rest = model_match.group("rest")

    parsed["model_name"] = f"{model_family}-{model_size}"

    seed_match = SEED_RE.search(rest)
    if seed_match is None:
        method_segment = rest
        extra_segment = None
    else:
        seed_index = seed_match.start()
        method_segment = rest[:seed_index]
        extra_segment = seed_match.group("suffix")
        parsed["seed"] = safe_int(seed_match.group("seed"))
        # keep extra_segment as a local variable (do not store in parsed)
        if extra_segment:
            extra_segment = extra_segment.lstrip("-") or None

    method_segment = method_segment.rstrip("-")
    method_match = METHOD_RE.match(method_segment)
    if method_match:
        parsed["method"] = method_match.group("method")
        rank_kind = method_match.group("rank_kind")
        rank_or_k = safe_int(method_match.group("rank_or_k"))
        if rank_kind and rank_or_k is not None:
            parsed["method_rank"] = f"{rank_kind}={rank_or_k}"
    else:
        parsed["method"] = method_segment or None
        # leave method as-is when unparsed; no parse_status field

    if extra_segment:
        alpha_match = ALPHA_RE.search(extra_segment)
        seq_match = SEQ_RE.search(extra_segment)
        plr_match = PLR_RE.search(extra_segment)
        if alpha_match:
            parsed["alpha"] = safe_float(alpha_match.group("alpha"))
        if seq_match:
            parsed["seq_len"] = safe_int(seq_match.group("seq_len"))
        if plr_match:
            parsed["plr"] = plr_match.group("plr")

    # Also try to parse parameters from the method variant for names like lora-preliminary.
    method_variant = method_match.group("variant") if method_match else None
    if method_variant:
        variant = method_variant
        alpha_match = ALPHA_RE.search(variant)
        seq_match = SEQ_RE.search(variant)
        plr_match = PLR_RE.search(variant)
        if parsed["alpha"] is None and alpha_match:
            parsed["alpha"] = safe_float(alpha_match.group("alpha"))
        if parsed["seq_len"] is None and seq_match:
            parsed["seq_len"] = safe_int(seq_match.group("seq_len"))
        if parsed["plr"] is None and plr_match:
            parsed["plr"] = plr_match.group("plr")

    return parsed


def collect_rows(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result_file in sorted(output_root.rglob("final_eval_results.jsonl")):
        run_dir = result_file.parent
        metadata = parse_run_name(run_dir, output_root)
        metrics = choose_metric_row(load_jsonl(result_file))

        row = {**metadata}
        for key in METRIC_KEYS:
            row[key] = scale_metric_value(metrics.get(key)) if key in SCALED_METRIC_KEYS else metrics.get(key)
        rows.append(row)

    return rows


def export_excel(rows: List[Dict[str, Any]], output_xlsx: Path) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    column_order = [
        "run_name",
        "model_name",
        "method",
        "method_rank",
        "seed",
        "alpha",
        "seq_len",
        "plr",
        # extra_suffix and parse_status removed
        *METRIC_KEYS,
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=column_order)
    else:
        for col in column_order:
            if col not in df.columns:
                df[col] = None
        df = df[column_order]

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_results")
        ws = writer.book["all_results"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        header_to_col = {cell.value: idx + 1 for idx, cell in enumerate(ws[1])}
        for col_name in METRIC_KEYS + ["alpha"]:
            col_idx = header_to_col.get(col_name)
            if col_idx is None:
                continue
            for cell in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                for item in cell:
                    if item.value is not None:
                        item.number_format = "0.00"

        for col_name in ["seed", "seq_len"]:
            col_idx = header_to_col.get(col_name)
            if col_idx is None:
                continue
            for cell in ws.iter_cols(min_col=col_idx, max_col=col_idx, min_row=2):
                for item in cell:
                    if item.value is not None:
                        item.number_format = "0"

        # Make the sheet easier to scan.
        for column_cells in ws.columns:
            values = [str(cell.value) for cell in column_cells if cell.value is not None]
            if not values:
                continue
            max_len = min(max(len(v) for v in values) + 2, 60)
            ws.column_dimensions[column_cells[0].column_letter].width = max_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final_eval_results.jsonl files into a single Excel table.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output_root", type=Path, default=repo_root / "output", help="Root directory containing experiment folders.")
    parser.add_argument("--output_xlsx", type=Path, default=repo_root / "all_results.xlsx", help="Destination Excel file.")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_xlsx = args.output_xlsx.resolve()

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    rows = collect_rows(output_root)
    export_excel(rows, output_xlsx)

    print(f"Saved {len(rows)} rows to {output_xlsx}")


if __name__ == "__main__":
    main()


