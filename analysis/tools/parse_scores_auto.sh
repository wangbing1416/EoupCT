#!/usr/bin/env bash
set -u

# Auto-parse pending output folders.
# A folder is processed only when:
#   - eval_results.jsonl exists
#   - mmlu_eval_results.json exists
#   - final_eval_results.jsonl does NOT exist

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="$REPO_ROOT/output"

if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "Output directory not found: $OUTPUT_ROOT"
  exit 0
fi

shopt -s nullglob

processed=0
skipped=0

while IFS= read -r -d '' run_dir; do
  eval_file="$run_dir/eval_results.jsonl"
  mmlu_file="$run_dir/mmlu_eval_results.json"
  final_file="$run_dir/final_eval_results.jsonl"

  if [[ -f "$eval_file" && -f "$mmlu_file" && ! -f "$final_file" ]]; then
    echo "[RUN] $run_dir"
    python "$REPO_ROOT/analysis/tools/parse_scores.py" --run_name "${run_dir#$REPO_ROOT/}"
    processed=$((processed + 1))
  else
    echo "[SKIP] $run_dir"
    skipped=$((skipped + 1))
  fi
done < <(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

echo "Done. processed=$processed skipped=$skipped"

