# RoupCT

This repository contains the training and evaluation code for two main paths:

- `run_loramoe.py`: the LoRAMoE / baseline training entrypoint
- `run_ours.py`: the RoupCT training entrypoint for our method

The `src/` folder contains batch shell scripts for running different model / seed / hyper-parameter sweeps.
The `analysis/` folder stores archive-only utilities and generated outputs.

## Installation

Create the environment with either Conda or pip:

```bash
conda env create -f environment.yml
```

or:

```bash
conda create -n roupct python=3.10 -y
pip install -r requirements.txt
```

> Note: the project uses the repository's local code layout, so do **not** install a separate upstream `peft` package unless you know you need it.

## Repository layout

- `run_loramoe.py` — baseline training / evaluation entrypoint
- `run_ours.py` — RoupCT training / evaluation entrypoint
- `src/` — batch launch scripts for baseline and our method
- `config/` — experiment config files
- `evaluation/` — MMLU evaluation helpers
- `compute_metrics.py` — EM / ROUGE-style scoring helpers
- `data_process.py` — dataset loading and tokenization
- `analysis/` — archived helpers, post-processing scripts, and results

## Data format

The training code expects SuperNI-style JSON files.

The launcher scripts use paths such as:

- `data/SuperNI`
- `data/superNI`
- `data/mmlu` for MMLU evaluation

The supervised JSON format used by `data_process.py` looks like this:

```json
{
  "Definition": "...",
  "Instances": [
    {
      "input": "...",
      "output": "..."
    }
  ]
}
```

If `output` is a list, one answer is sampled during dataset building.

## How to run

### 1) Baseline / LoRAMoE

The main baseline launcher is:

```bash
bash src/run_loramoe.sh
```

Each model-specific wrapper in `src/run_lora_*.sh` shares the same calling pattern:

```bash
bash src/run_lora_qwen3-4b.sh <seed> <config_file> <run_name>
```

Examples:

```bash
bash src/run_lora_qwen3-4b.sh 1 config/lora.config lora-r4
bash src/run_lora_llama3-8b.sh 3 config/loramoe_k8.config loramoe-k8
```

The common launcher parameters include:

- `seed`
- model/config path
- `lora_config_file`
- `--deepspeed ds_zero2_no_offload.json`
- `--flash_attn` and `--attn_implementation flash_attention_2`

### 2) Our method / RoupCT

The main launcher for our method is:

```bash
bash src/run_ours.sh
```

For a single run, call the model-specific script directly. The argument order is:

```bash
bash src/run_ours_qwen3-4b.sh <seed> <config_file> <run_name> <roupct_alpha> <roupct_seq_len> <roupct_soft_prompt_lr> [tau] [lambda_pen]
```

Example:

```bash
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.05 16 1e-3
```

The RoupCT launcher appends these extra flags to `run_ours.py`:

- `--roupct_alpha`
- `--roupct_seq_len`
- `--roupct_soft_prompt_lr`
- `--roupct_tau`
- `--roupct_lambda_pen`

## Evaluation

Evaluation is handled inside the training scripts and the helper modules under `evaluation/`.

### Task-level metrics

`compute_metrics.py` provides the exact-match / ROUGE-style scoring used by the training scripts.

### MMLU evaluation

`evaluation/mmlu_runner.py` implements MMLU evaluation and is used by both `run_loramoe.py` and `run_ours.py`.

Relevant MMLU-related arguments include:

- `mmlu_eval_after_each_task`
- `mmlu_test_dir` (default: `data/mmlu`)
- `mmlu_subjects`
- `mmlu_ntrain`
- `mmlu_max_length`
- `mmlu_batch_size`

### Output folders

Training outputs are written under `./output/<exp_name>/`.

If you need to parse saved metrics after a run, the archived tools in `analysis/tools/` are still available.

## Useful scripts

- `bash src/run_loramoe.sh` — main LoRAMoE sweep launcher
- `bash src/run_lora.sh` — LoRA baseline sweep launcher
- `bash src/run_ours.sh` — main RoupCT sweep launcher
- `bash src/run_preliminary.sh` — smaller preliminary subset
- `bash src/run_pissa+olora.sh` — PiSSA / OLoRA comparison runs

## Archived files

The following items are now archive-only and are no longer part of the main runtime path:

- `analysis/legacy/` — legacy helpers and old patch code
- `analysis/tools/` — post-processing / visualization scripts
- `analysis/results/` — generated metric files and result snapshots

## Notes

- The repository keeps a local `transformers/` tree; do not replace it unless you are intentionally changing the model runtime.
- Set `MODEL_DIR` in the shell launchers to your local checkpoint directory before running.
- `run_loramoe.py` and `run_ours.py` both support `--do_train` / `--do_eval` workflows with DeepSpeed.

## Citation

If you find this useful in your research, please consider citing the associated paper or project report.
