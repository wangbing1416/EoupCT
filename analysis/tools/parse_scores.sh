#!/usr/bin/env bash
# Auto-generated parse_scores.sh - runs parse_scores.py for each output folder
# Generated to match the list you provided; folders containing "preliminary" are excluded.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-loramoe-k4-seed1
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-loramoe-k4-seed2
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-loramoe-k4-seed3
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-loramoe-k8-seed1
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-loramoe-k8-seed2
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-loramoe-k8-seed3
#
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-lora-r4-seed1
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-lora-r4-seed2
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-lora-r4-seed3
#
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-olora-r4-seed1
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-olora-r4-seed2
#python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-olora-r4-seed3

python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.02-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.02-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.02-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.05-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.05-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.05-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.1-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.1-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed1-a0.1-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.02-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.02-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.02-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.05-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.05-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.05-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.1-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.1-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed2-a0.1-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.02-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.02-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.02-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.05-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.05-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.05-s8-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.1-s16-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.1-s4-plr1e-3
python "$SCRIPT_DIR/parse_scores.py" --run_name output/gemma2-2b-ours-seed3-a0.1-s8-plr1e-3

# gemma2-9b ours entries
# ...existing code...

#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-2b-pissa-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-2b-pissa-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-2b-pissa-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-loramoe-k4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-loramoe-k4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-loramoe-k4-seed3
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-loramoe-k8-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-loramoe-k8-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-loramoe-k8-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-lora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-lora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-lora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-olora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-olora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-olora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-pissa-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-pissa-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/gemma2-9b-pissa-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-loramoe-k4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-loramoe-k4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-loramoe-k4-seed3
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-loramoe-k8-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-loramoe-k8-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-loramoe-k8-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-lora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-lora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-lora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-olora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-olora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-olora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-pissa-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-pissa-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-3b-pissa-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-loramoe-k4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-loramoe-k4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-loramoe-k4-seed3
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-loramoe-k8-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-loramoe-k8-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-loramoe-k8-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-lora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-lora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-lora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-olora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-olora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-olora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-pissa-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-pissa-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/llama3-8b-pissa-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-loramoe-k4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-loramoe-k4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-loramoe-k4-seed3
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-loramoe-k8-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-loramoe-k8-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-loramoe-k8-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-lora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-lora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-lora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-olora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-olora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-olora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-pissa-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-pissa-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-4b-pissa-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-loramoe-k4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-loramoe-k4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-loramoe-k4-seed3
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-loramoe-k8-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-loramoe-k8-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-loramoe-k8-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-lora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-lora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-lora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-olora-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-olora-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-olora-r4-seed3
#
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-pissa-r4-seed1
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-pissa-r4-seed2
#python $SCRIPT_DIR/parse_scores.py --run_name output/qwen3-8b-pissa-r4-seed3

