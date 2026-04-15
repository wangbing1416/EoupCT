import os
import json
import re

# Load JSON lines.
def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def export_colored_latex_table(a, b, output_file='output.tex', color_name='tomato'):
    """Convert a 2D numeric list into LaTeX table rows.

    Each value is formatted to one decimal place and wrapped in a colored
    `\cellcolor{color_name!xx.xx}xx.xx` cell. Each row is prefixed with the
    corresponding element from `b`, joined with `&`, and written to a .tex file.
    """
    if len(a) != len(b):
        raise ValueError("The number of rows in a and b must be the same.")

    lines = []
    for row_a, prefix in zip(a, b):
        # Format each number as a string with one decimal place.
        formatted_vals = [f"{val:.1f}" for val in row_a]
        # Build colored LaTeX cells.
        colored_cells = [f"\\cellcolor{{{color_name}!{val}}}{val}" for val in formatted_vals]
        # Concatenate the prefix and data row.
        full_row = [str(prefix)] + colored_cells
        line = " & ".join(full_row) + " \\\\"
        lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved to {output_file}")


def cal_continue_learning_metrics(scores_array, baseline_array):
    task_num = len(scores_array)

    Cl = sum(scores_array[-1]) / task_num

    fgt_list = []
    for t_idx in range(task_num - 1):
        history = [line[t_idx] for line in scores_array[:-1]]
        history_best = max(history)
        fgt_list.append(history_best - scores_array[-1][t_idx])
    Fgt = sum(fgt_list) / len(fgt_list)

    if len(baseline_array) > 0:
        Fwt = sum([scores_array[i][i] for i in range(task_num)]) / task_num - sum(baseline_array) / task_num
    else:
        Fwt = 0

    Bwt = sum([scores_array[-1][i] - scores_array[i][i] for i in range(task_num)]) / task_num

    return {
        'Cl': Cl,
        'Fgt': Fgt,
        'Fwt': Fwt,
        'Bwt': Bwt,
    }


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='output/gemma2-9b-loramoe-seed3')
    args = parser.parse_args()

    task_order = load_json(f'./{args.run_name}/task_order.json')[0]
    task_num = len(task_order)
    results = load_json(f'./{args.run_name}/eval_results.jsonl')
    results = sorted(results, key=lambda x: task_order.index(x['training_task_name']))

    baseline_root = f"{args.run_name.split('-seed')[0].split('-r4')[0]}-baseline"
    try:
        baselines = [load_json(f"./{baseline_root}/eval_results_{task}.jsonl") for task in task_order]
    except:
        baselines = [None for task in task_order]

    prefix = 'eval_rougeL_for_'
    # filter the useful values
    filtered_results = []
    baseline_results = []
    for t, res, basel in zip(task_order, results, baselines):
        fil_result = {}
        for task_i, task in enumerate(task_order):
            fil_result[prefix + task] = res[prefix + task]
        filtered_results.append(fil_result)
        if basel:
            baseline_results.append(basel[0][prefix + t])

    scores = []
    for task_i, task in enumerate(task_order):
        scores.append(list(filtered_results[task_i].values()))

    export_colored_latex_table(scores, task_order, f'./{args.run_name}/continual_learning_scores.tex')

    metrics = cal_continue_learning_metrics(scores, baseline_results)


    def parse_mmlu_file(mmlu_path):
        """Helper to parse a single mmlu_eval_results.json"""
        local_metrics = {}
        if os.path.exists(mmlu_path):
            try:
                with open(mmlu_path, 'r', encoding='utf-8') as f_mmlu:
                    mmlu_data = json.load(f_mmlu)

                if mmlu_data:
                    # Logic assumes list of records
                    mmlu_history = mmlu_data if isinstance(mmlu_data, list) else [mmlu_data]
                    if mmlu_history:
                        # Get category_acc from first and last results
                        first_res_cats = mmlu_history[0].get('category_acc', {})
                        last_res_cats = mmlu_history[-1].get('category_acc', {})

                        # Also record overall weighted accuracy
                        local_metrics['mmlu_weighted_acc'] = mmlu_history[-1].get('mmlu_weighted_acc', 0)

                        for cat, acc in last_res_cats.items():
                            local_metrics[f'{cat}_acc'] = acc
                            if cat in first_res_cats:
                                # Forgetting = First - Last (Positive means performance loss)
                                local_metrics[f'{cat}_forgetting'] = first_res_cats[cat] - acc
            except Exception as e:
                print(f"Error parsing mmlu results {mmlu_path}: {e}")
        return local_metrics

    mmlu_metrics = {}

    # Identify seed directories: match the same hyper-parameter suffix, only varying the seed number.
    run_path_clean = args.run_name.rstrip('/\\')
    dir_name = os.path.dirname(run_path_clean)
    base_name = os.path.basename(run_path_clean)

    seed_dirs = []
    seed_match = re.match(r'^(?P<prefix>.*-seed)(?P<seed>\d+)(?P<suffix>.*)$', base_name)
    if seed_match:
        # Example:
        #   gemma2-2b-ours-seed1-a0.02-s16-plr1e-3
        #   -> prefix: gemma2-2b-ours-seed
        #   -> suffix: -a0.02-s16-plr1e-3
        prefix = seed_match.group('prefix')
        suffix = seed_match.group('suffix')
        search_root = dir_name if dir_name else '.'

        candidates = []
        for entry in os.listdir(search_root):
            full_path = os.path.join(search_root, entry)
            if not os.path.isdir(full_path):
                continue

            entry_match = re.match(r'^(?P<prefix>.*-seed)(?P<seed>\d+)(?P<suffix>.*)$', entry)
            if not entry_match:
                continue

            if entry_match.group('prefix') == prefix and entry_match.group('suffix') == suffix:
                candidates.append((int(entry_match.group('seed')), full_path))

        seed_dirs = [path for _, path in sorted(candidates, key=lambda x: (x[0], x[1]))]

    # Fallback to current dir if no seed pattern found or empty results
    if not seed_dirs:
        seed_dirs = [args.run_name]

    print(f"Aggregating MMLU metrics from {len(seed_dirs)} directories: {[os.path.basename(d) for d in seed_dirs]}")

    # Collect and average metrics
    sums = {}
    counts = {}

    for d in seed_dirs:
        path = os.path.join(d, 'mmlu_eval_results.json')
        m_res = parse_mmlu_file(path)
        for k, v in m_res.items():
            if k not in sums:
                sums[k] = 0.0
                counts[k] = 0
            sums[k] += v
            counts[k] += 1

    for k in sums:
        mmlu_metrics[k] = sums[k] / counts[k] if counts[k] > 0 else 0

    output_path = f"./{args.run_name}/final_eval_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if mmlu_metrics:
            f.write(json.dumps(mmlu_metrics, ensure_ascii=False) + "\n")
        for res in filtered_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # Format metrics: keep original scale but round to 2 decimals
    metrics_formatted = {k: round(v, 2) for k, v in metrics.items()}

    # Format MMLU metrics: scale by 100 and round to 2 decimals
    mmlu_metrics_pct = {k: round(v * 100, 2) for k, v in mmlu_metrics.items()}

    print(f"**********\n"
          f"Final eval metrics: {json.dumps(metrics_formatted, ensure_ascii=False, sort_keys=False)} \n"
          f"MMLU metrics: {json.dumps(mmlu_metrics_pct, ensure_ascii=False, sort_keys=False) if mmlu_metrics_pct else 'N/A'} \n"
          f"the file has been saved in {output_path} \n"
          f"**********\n")
