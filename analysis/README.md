# Analysis Archive

This folder collects files that are not part of the main training/runtime path.

## Kept at repository root
- `run_loramoe.py` — LoRAMoE baseline training entrypoint
- `run_ours.py` — our RoupCT training entrypoint
- `src/*.sh` — batch launch scripts for training runs
- `data_process.py`, `build_trainer.py`, `compute_metrics.py`, `roupct_trainer.py`, `evaluation/` — shared runtime dependencies

## Archived here

### `analysis/legacy/`
Unused or legacy helpers that are not imported by the current entrypoints:
- `build_dataset.py`
- `dataset_utils.py`
- `utils.py`
- `flash_attn_patch.py`
- `peft_class.py`

### `analysis/tools/`
Analysis / post-processing scripts:
- `collect_all_results.py`
- `parse_scores.py`
- `parse_scores.sh`
- `parse_scores_auto.sh`
- `visualize_lora_weights.py`
- `visualize_scores.py`

### `analysis/results/`
Generated experiment outputs and intermediate result files:
- `eval_results.jsonl`
- `final_eval_results.jsonl`
- `mmlu_eval_results.json`
- `weight_search_results.json`

## Notes
- I intentionally kept the training launchers and configs in place so the main workflows remain runnable.
- The archived shell scripts in `analysis/tools/` now point to their relocated Python companion script.

