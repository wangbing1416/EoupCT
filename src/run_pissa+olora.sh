#!/bin/bash

# Define all scripts to run.

## Iterate over each script.
#for seed in 1 2 3; do
#    echo "---- start running run_lora_gemma2-2b.sh, seed: $seed"
#    bash src/run_lora_gemma2-2b.sh "$seed" config/olora.config olora-r4
#    bash src/run_lora_gemma2-2b.sh "$seed" config/pissa.config pissa-r4
#done

#for seed in 1 2 3; do
for seed in 3; do
    echo "---- start running run_lora_llama3-3b.sh, seed: $seed"
    bash src/run_lora_llama3-3b.sh "$seed" config/olora.config olora-r4
    bash src/run_lora_llama3-3b.sh "$seed" config/pissa.config pissa-r4
done

for seed in 1 2 3; do
    echo "---- start running /run_lora_qwen3-4b.sh, seed: $seed"
    bash src/run_lora_qwen3-4b.sh "$seed" config/olora.config olora-r4
    bash src/run_lora_qwen3-4b.sh "$seed" config/pissa.config pissa-r4
done

for seed in 1 2 3; do
    echo "---- start running run_lora_llama3-8b.sh, seed: $seed"
    bash src/run_lora_llama3-8b.sh "$seed" config/olora.config olora-r4
    bash src/run_lora_llama3-8b.sh "$seed" config/pissa.config pissa-r4
done

for seed in 1 2 3; do
    echo "---- start running /run_lora_qwen3-8b.sh, seed: $seed"
    bash src/run_lora_qwen3-8b.sh "$seed" config/olora.config olora-r4
    bash src/run_lora_qwen3-8b.sh "$seed" config/pissa.config pissa-r4
done

for seed in 1 2 3; do
    echo "---- start running /run_lora_gemma2-9b.sh, seed: $seed"
    bash src/run_lora_gemma2-9b.sh "$seed" config/olora.config olora-r4
    bash src/run_lora_gemma2-9b.sh "$seed" config/pissa.config pissa-r4
done