"""Convert MMLU MoE weights JSON into an Excel table and a heatmap image.

Usage:
    python -m evaluation.collect_mmlu_results_icml path/to/weights.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import AxesImage


WEIGHT_COUNT = 8
WEIGHT_COLUMNS = [f"weight_{i}" for i in range(1, WEIGHT_COUNT + 1)]


def _validate_weights(payload: object) -> Dict[str, List[float]]:
    if not isinstance(payload, dict):
        raise ValueError("The top-level JSON must be a dictionary in the form {task_name: [8 weights]}.")

    normalized: Dict[str, List[float]] = {}
    for task_name, values in payload.items():
        if not isinstance(task_name, str):
            raise ValueError(f"Task names must be strings, but got: {task_name!r}")
        if not isinstance(values, list):
            raise ValueError(f"Task {task_name!r} must map to a list of length 8.")
        if len(values) != WEIGHT_COUNT:
            raise ValueError(
                f"Task {task_name!r} must contain {WEIGHT_COUNT} weights, but got {len(values)}."
            )

        row: List[float] = []
        for idx, value in enumerate(values, start=1):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Task {task_name!r} has a non-numeric weight at position {idx}: {value!r}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"Task {task_name!r} has a non-finite weight at position {idx}: {value!r}"
                )
            row.append(value)
        normalized[task_name] = row

    if not normalized:
        raise ValueError("JSON cannot be empty; at least one task is required.")
    return normalized


def _load_json(json_path: Path) -> Dict[str, List[float]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _validate_weights(payload)


def _build_dataframe(weights: Dict[str, List[float]]) -> pd.DataFrame:
    rows = []
    for task_name, values in weights.items():
        row: Dict[str, Union[str, float]] = {"task": task_name}
        for col, value in zip(WEIGHT_COLUMNS, values):
            row[col] = round(value, 2)
        rows.append(row)
    return pd.DataFrame(rows, columns=["task"] + WEIGHT_COLUMNS)


def _build_topk_binary_dataframe(weights: Dict[str, List[float]], top_k: int) -> pd.DataFrame:
    rows = []
    for task_name, values in weights.items():
        arr = np.asarray(values, dtype=float)
        mask = np.zeros_like(arr, dtype=int)
        top_indices = np.argsort(-arr, kind="stable")[:top_k]
        mask[top_indices] = 1

        row: Dict[str, Union[str, int]] = {"task": task_name}
        for col, value in zip(WEIGHT_COLUMNS, mask.tolist()):
            row[col] = int(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=["task"] + WEIGHT_COLUMNS)


def _build_sharpen_dataframe(weights: Dict[str, List[float]], alpha: float) -> pd.DataFrame:
    rows = []
    for task_name, values in weights.items():
        arr = np.asarray(values, dtype=float)
        clipped = np.clip(arr, a_min=0.0, a_max=None)
        if not np.any(clipped):
            probs = np.full_like(clipped, 1.0 / len(clipped), dtype=float)
        else:
            powered = np.power(clipped, alpha)
            denom = float(np.sum(powered))
            if denom <= 0 or not math.isfinite(denom):
                probs = np.full_like(clipped, 1.0 / len(clipped), dtype=float)
            else:
                probs = powered / denom

        row: Dict[str, Union[str, float]] = {"task": task_name}
        for col, value in zip(WEIGHT_COLUMNS, probs.tolist()):
            row[col] = float(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=["task"] + WEIGHT_COLUMNS)


def _save_excel(
    df: pd.DataFrame,
    binary_df: pd.DataFrame,
    sharpen_df: pd.DataFrame,
    excel_path: Path,
    top_k: int,
    alpha: float,
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="weights")
        binary_sheet_name = f"top{top_k}_binary"
        binary_df.to_excel(writer, index=False, sheet_name=binary_sheet_name)
        sharpen_df.to_excel(writer, index=False, sheet_name="sharpen")

        ws = writer.book["weights"]
        for col_cells in ws.iter_cols(min_col=2, max_col=1 + WEIGHT_COUNT, min_row=2):
            for cell in col_cells:
                cell.number_format = "0.00"
        ws.freeze_panes = "B2"

        ws_binary = writer.book[binary_sheet_name]
        for col_cells in ws_binary.iter_cols(min_col=2, max_col=1 + WEIGHT_COUNT, min_row=2):
            for cell in col_cells:
                cell.number_format = "0"
        ws_binary.freeze_panes = "B2"

        ws_sharpen = writer.book["sharpen"]
        for col_cells in ws_sharpen.iter_cols(min_col=2, max_col=1 + WEIGHT_COUNT, min_row=2):
            for cell in col_cells:
                cell.number_format = "0.00"
        ws_sharpen.freeze_panes = "B2"


def _render_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    value_fmt: str,
    xlabel: str,
) -> AxesImage:
    values = df[WEIGHT_COLUMNS].to_numpy(dtype=float)
    rows = len(df)
    cols = len(WEIGHT_COLUMNS)

    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(cols))
    ax.set_xticklabels(WEIGHT_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(range(rows))
    ax.set_yticklabels(df["task"].tolist())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MMLU task")
    ax.set_title(title)

    for i in range(rows):
        for j in range(cols):
            ax.text(
                j,
                i,
                value_fmt.format(values[i, j]),
                ha="center",
                va="center",
                color="#000000",
                fontsize=8,
            )

    return im


def _make_combined_heatmap(
    raw_df: pd.DataFrame,
    binary_df: pd.DataFrame,
    sharpen_df: pd.DataFrame,
    image_path: Path,
    top_k: int,
    alpha: float,
) -> None:
    raw_values = raw_df[WEIGHT_COLUMNS].to_numpy(dtype=float)
    raw_vmin = float(np.min(raw_values))
    raw_vmax = float(np.max(raw_values))
    if math.isclose(raw_vmin, raw_vmax):
        raw_vmax = raw_vmin + 1e-9

    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#ff0000"])
    fig_width = max(22.0, 3.0 + len(WEIGHT_COLUMNS) * 1.2)
    fig_height = max(4.0, 0.45 * len(raw_df) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=200)
    axes_list = list(np.atleast_1d(axes).ravel())

    raw_im = _render_heatmap(
        axes_list[0],
        raw_df,
        "Original MoE Weights",
        cmap,
        raw_vmin,
        raw_vmax,
        "{:.2f}",
        "MoE weights",
    )
    binary_im = _render_heatmap(
        axes_list[1],
        binary_df,
        f"Top-{top_k} Binary Mask",
        cmap,
        0.0,
        1.0,
        "{:.0f}",
        f"Top-{top_k} mask",
    )

    sharpen_im = _render_heatmap(
        axes_list[2],
        sharpen_df,
        f"Sharpened Weights (alpha={alpha:g})",
        cmap,
        0.0,
        1.0,
        "{:.2f}",
        f"Sharpen alpha={alpha:g}",
    )

    fig.colorbar(raw_im, ax=axes_list[0], fraction=0.046, pad=0.04)
    fig.colorbar(binary_im, ax=axes_list[1], fraction=0.046, pad=0.04)
    fig.colorbar(sharpen_im, ax=axes_list[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def export_weights(json_path: Path) -> Tuple[Path, Path]:
    return export_weights_with_topk(json_path, top_k=4, alpha=2.0)


def export_weights_with_topk(json_path: Path, top_k: int, alpha: float) -> Tuple[Path, Path]:
    weights = _load_json(json_path)
    df = _build_dataframe(weights)
    binary_df = _build_topk_binary_dataframe(weights, top_k=top_k)
    sharpen_df = _build_sharpen_dataframe(weights, alpha=alpha)

    excel_path = json_path.with_suffix(".xlsx")
    image_path = json_path.with_suffix(".png")

    _save_excel(df, binary_df, sharpen_df, excel_path, top_k=top_k, alpha=alpha)
    _make_combined_heatmap(df, binary_df, sharpen_df, image_path, top_k=top_k, alpha=alpha)
    return excel_path, image_path


def _parse_top_k(value: str) -> int:
    top_k = int(value)
    if top_k < 1 or top_k > WEIGHT_COUNT:
        raise argparse.ArgumentTypeError(f"top_k must be between 1 and {WEIGHT_COUNT}.")
    return top_k


def _parse_alpha(value: str) -> float:
    alpha = float(value)
    if alpha <= 0:
        raise argparse.ArgumentTypeError("sharpen_alpha must be greater than 0.")
    return alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MMLU MoE weights JSON into an Excel file and a heatmap image."
    )
    parser.add_argument("--json_file", type=Path,
                        default=Path("<MMLU_WEIGHTS_JSON>"),
                        help="Path to the input JSON file.")
    parser.add_argument(
        "--top_k",
        "--top-k",
        dest="top_k",
        type=_parse_top_k,
        default=4,
        help=f"Set the top K weights for each task to 1 (1~{WEIGHT_COUNT}, default: 4).",
    )
    parser.add_argument(
        "--sharpen_alpha",
        "--sharpen-alpha",
        dest="sharpen_alpha",
        type=_parse_alpha,
        default=2.0,
        help="Sharpening strength; larger values produce a sharper distribution. Default: 2.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.json_file.expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if json_path.suffix.lower() != ".json":
        raise ValueError(f"The input file must be a JSON file: {json_path}")

    excel_path, image_path = export_weights_with_topk(
        json_path,
        top_k=args.top_k,
        alpha=args.sharpen_alpha,
    )
    print(f"Excel saved to: {excel_path}")
    print(f"Image saved to: {image_path}")


if __name__ == "__main__":
    main()



