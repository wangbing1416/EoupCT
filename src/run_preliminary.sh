#!/bin/bash

# Run the preliminary scripts for the six models with seed=1.

seed=1

#echo "---- start running run_lora_gemma2-2b_preliminary.sh, seed: $seed"
#bash src/run_lora_gemma2-2b_preliminary.sh "$seed" config/lora.config lora-preliminary
#
#echo "---- start running run_lora_llama3-3b_preliminary.sh, seed: $seed"
#bash src/run_lora_llama3-3b_preliminary.sh "$seed" config/lora.config lora-preliminary
#
#echo "---- start running run_lora_qwen3-4b_preliminary.sh, seed: $seed"
#bash src/run_lora_qwen3-4b_preliminary.sh "$seed" config/lora.config lora-preliminary
#
#echo "---- start running run_lora_llama3-8b_preliminary.sh, seed: $seed"
#bash src/run_lora_llama3-8b_preliminary.sh "$seed" config/lora.config lora-preliminary

echo "---- start running run_lora_qwen3-8b_preliminary.sh, seed: $seed"
bash src/run_lora_qwen3-8b_preliminary.sh "$seed" config/lora.config lora-preliminary

echo "---- start running run_lora_gemma2-9b_preliminary.sh, seed: $seed"
bash src/run_lora_gemma2-9b_preliminary.sh "$seed" config/lora.config lora-preliminary

