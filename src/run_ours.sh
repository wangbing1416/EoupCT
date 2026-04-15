#!/usr/bin/env bash
set -euo pipefail

# Manual run list for RoupCT experiments.
# Comment out any line you do not want to run.

# gemma2-2b
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.02 4
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.02 8
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.02 16
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.05 4
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.05 8
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.05 16
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.1 4
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.1 8
#bash src/run_ours_gemma2-2b.sh 1 config/lora.config ours 0.1 16
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.02 4
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.02 8
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.02 16
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.05 4
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.05 8
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.05 16
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.1 4
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.1 8
#bash src/run_ours_gemma2-2b.sh 2 config/lora.config ours 0.1 16
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.02 4
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.02 8
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.02 16
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.05 4
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.05 8
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.05 16
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.1 4
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.1 8
#bash src/run_ours_gemma2-2b.sh 3 config/lora.config ours 0.1 16
#
## gemma2-9b
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.02 4
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.02 8
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.02 16
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.05 4
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.05 8
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.05 16
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.1 4
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.1 8
#bash src/run_ours_gemma2-9b.sh 1 config/lora.config ours 0.1 16
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.02 4
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.02 8
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.02 16
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.05 4
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.05 8
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.05 16
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.1 4
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.1 8
#bash src/run_ours_gemma2-9b.sh 2 config/lora.config ours 0.1 16
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.02 4
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.02 8
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.02 16
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.05 4
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.05 8
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.05 16
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.1 4
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.1 8
#bash src/run_ours_gemma2-9b.sh 3 config/lora.config ours 0.1 16
#
## llama3-3b
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.02 4
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.02 8
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.02 16
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.05 4
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.05 8
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.05 16
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.1 4
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.1 8
#bash src/run_ours_llama3-3b.sh 1 config/lora.config ours 0.1 16
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.02 4
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.02 8
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.02 16
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.05 4
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.05 8
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.05 16
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.1 4
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.1 8
#bash src/run_ours_llama3-3b.sh 2 config/lora.config ours 0.1 16
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.02 4
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.02 8
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.02 16
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.05 4
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.05 8
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.05 16
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.1 4
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.1 8
#bash src/run_ours_llama3-3b.sh 3 config/lora.config ours 0.1 16

# llama3-8b
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.02 4
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.02 8
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.02 16
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.05 4
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.05 8
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.05 16
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.1 4
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.1 8
#bash src/run_ours_llama3-8b.sh 1 config/lora.config ours 0.1 16
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.02 4
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.02 8
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.02 16
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.05 4
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.05 8
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.05 16
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.1 4
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.1 8
#bash src/run_ours_llama3-8b.sh 2 config/lora.config ours 0.1 16
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.02 4
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.02 8
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.02 16
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.05 4
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.05 8
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.05 16
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.1 4
#bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.1 8
bash src/run_ours_llama3-8b.sh 3 config/lora.config ours 0.1 16

# qwen3-4b
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.02 4
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.02 8
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.02 16
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.05 4
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.05 8
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.05 16
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.1 4
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.1 8
bash src/run_ours_qwen3-4b.sh 1 config/lora.config ours 0.1 16
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.02 4
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.02 8
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.02 16
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.05 4
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.05 8
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.05 16
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.1 4
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.1 8
bash src/run_ours_qwen3-4b.sh 2 config/lora.config ours 0.1 16
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.02 4
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.02 8
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.02 16
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.05 4
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.05 8
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.05 16
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.1 4
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.1 8
bash src/run_ours_qwen3-4b.sh 3 config/lora.config ours 0.1 16

# qwen3-8b
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.02 4
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.02 8
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.02 16
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.05 4
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.05 8
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.05 16
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.1 4
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.1 8
bash src/run_ours_qwen3-8b.sh 1 config/lora.config ours 0.1 16
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.02 4
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.02 8
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.02 16
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.05 4
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.05 8
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.05 16
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.1 4
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.1 8
bash src/run_ours_qwen3-8b.sh 2 config/lora.config ours 0.1 16
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.02 4
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.02 8
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.02 16
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.05 4
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.05 8
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.05 16
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.1 4
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.1 8
bash src/run_ours_qwen3-8b.sh 3 config/lora.config ours 0.1 16



