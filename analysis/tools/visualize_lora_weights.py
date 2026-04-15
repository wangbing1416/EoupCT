import os
import sys
import json
import argparse
import importlib


CATEGORY_COLUMNS = [
    ("STEM", "STEM_acc"),
    ("humanities", "humanities_acc"),
    ("other (business, health, misc.)", "other_acc"),
]


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_rows(weight_results):
    rows = []
    for item in weight_results:
        eval_metrics = item.get('eval_metrics', {}) or {}
        mmlu_metrics = item.get('mmlu_metrics', {}) or {}
        category_acc = mmlu_metrics.get('category_acc', {}) or {}

        row = {
            'weight': as_float(item.get('weight')),
            'eval_rougeL': as_float(eval_metrics.get('eval_rougeL')),
        }

        category_values = []
        for category_key, column_name in CATEGORY_COLUMNS:
            value = as_float(category_acc.get(category_key))
            row[column_name] = value
            if value is not None:
                category_values.append(value)

        row['category_acc_mean'] = sum(category_values) / len(category_values) if category_values else None
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='output/gemma2-9b-loramoe-seed3',
                        help='Path or run name of the directory containing weight_search_results.json')
    args = parser.parse_args()

    run_name = args.run_name.rstrip('/\\')

    try:
        pd = importlib.import_module('pandas')
        plt = importlib.import_module('matplotlib.pyplot')
    except ImportError as e:
        print(f"Error: Required dependency not found. Please run 'pip install pandas matplotlib openpyxl'.\nDetail: {e}")
        sys.exit(1)

    input_path = os.path.join(run_name, 'weight_search_results.json')
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    try:
        weight_results = load_json(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        sys.exit(1)

    if isinstance(weight_results, dict):
        weight_results = [weight_results]
    elif not isinstance(weight_results, list):
        print(f"Error: {input_path} must contain a JSON array or object.")
        sys.exit(1)

    rows = build_rows(weight_results)
    if not rows:
        print(f"Error: no valid records found in {input_path}.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df = df.sort_values('weight', kind='mergesort').reset_index(drop=True)

    out_dir = './visualization'
    os.makedirs(out_dir, exist_ok=True)

    run_base = os.path.basename(run_name)
    excel_path = os.path.join(out_dir, f'{run_base}_lora_weights.xlsx')
    plot_path = os.path.join(out_dir, f'{run_base}_lora_weights.png')

    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='weight_search', index=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x = df['weight']
    ax1.plot(x, df['eval_rougeL'], marker='o', linewidth=2.2, label='eval_rougeL', color='#1f77b4')

    category_colors = ['#d62728', '#2ca02c', '#ff7f0e']
    for (category_key, column_name), color in zip(CATEGORY_COLUMNS, category_colors):
        ax2.plot(x, df[column_name], marker='s', linewidth=1.8, label=column_name, color=color)

    ax2.plot(x, df['category_acc_mean'], marker='^', linewidth=2.2, linestyle='--', label='category_acc_mean', color='#9467bd')

    ax1.set_xlabel('weight')
    ax1.set_ylabel('eval_rougeL')
    ax2.set_ylabel('category_acc')
    ax1.set_title('LoRA weight search metrics progression')
    ax1.grid(True, axis='both', alpha=0.3)

    ax1.tick_params(axis='x', rotation=45)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'Excel file saved to {excel_path}')
    print(f'Plot saved to {plot_path}')


if __name__ == '__main__':
    main()

