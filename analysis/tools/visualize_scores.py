import os
import sys
import json
import argparse
import importlib

def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='output/gemma2-9b-lora-r4-seed1',
                        help="Path or run name of the directory containing the json files")
    args = parser.parse_args()

    run_name = args.run_name.rstrip('/\\')
    
    # Check dependencies before trying to import to provide better error messages
    try:
        pd = importlib.import_module('pandas')
        plt = importlib.import_module('matplotlib.pyplot')
    except ImportError as e:
        print(f"Error: Required dependency not found. Please run 'pip install pandas matplotlib openpyxl'.\nDetail: {e}")
        sys.exit(1)
        
    eval_results_path = os.path.join(run_name, 'eval_results.jsonl')
    mmlu_results_path = os.path.join(run_name, 'mmlu_eval_results.json')
    task_order_path = os.path.join(run_name, 'task_order.json')
    
    if not os.path.exists(task_order_path):
        print(f"Error: {task_order_path} not found.")
        sys.exit(1)
        
    task_order = load_json(task_order_path)[0]
    
    # Process eval_results
    if not os.path.exists(eval_results_path):
        print(f"Error: {eval_results_path} not found.")
        sys.exit(1)
        
    results = load_json(eval_results_path)
    # sorted by training task order
    results = sorted(results, key=lambda x: task_order.index(x['training_task_name']))
    
    prefix = 'eval_rougeL_for_'
    eval_matrix = []
    for res in results:
        row = {}
        row['training_task'] = res.get('training_task_name', 'unknown')
        for task in task_order:
            row[task] = res.get(prefix + task, None)
        eval_matrix.append(row)
        
    df_eval = pd.DataFrame(eval_matrix)
    
    # Process MMLU results
    if not os.path.exists(mmlu_results_path):
        print(f"Error: {mmlu_results_path} not found.")
        sys.exit(1)
        
    with open(mmlu_results_path, 'r', encoding='utf-8') as f:
        mmlu_data = json.load(f)
        
    mmlu_matrix = []
    subjects = []
    if len(mmlu_data) > 0:
        subjects = list(mmlu_data[0].get('subject_acc', {}).keys())
        
    for i, data in enumerate(mmlu_data):
        row = {}
        row['training_task'] = task_order[i] if i < len(task_order) else f"step_{i+1}"
        subject_acc = data.get('subject_acc', {})
        for sub in subjects:
            row[sub] = subject_acc.get(sub, None)
        subject_values = [v for v in subject_acc.values() if isinstance(v, (int, float))]
        row['subject_avg_acc'] = sum(subject_values) / len(subject_values) if subject_values else None
        mmlu_matrix.append(row)
        
    df_mmlu = pd.DataFrame(mmlu_matrix)
    
    # Output to Excel
    out_dir = './visualization'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    run_base = os.path.basename(run_name)
    excel_path = os.path.join(out_dir, f"{run_base}_scores.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        df_eval.to_excel(writer, sheet_name='eval_rougeL', index=False)
        df_mmlu.to_excel(writer, sheet_name='mmlu_subject_acc', index=False)
        
    print(f"Excel file saved to {excel_path}")
    
    # Output to Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x_labels_eval = df_eval['training_task'].tolist()
    for task in task_order:
        ax1.plot(df_eval['training_task'], df_eval[task], marker='o', label=task)
    ax1.set_title('eval_rougeL Performance Progression')
    ax1.set_xlabel('Training Progression (Assessed After Task)')
    ax1.set_ylabel('eval_rougeL')
    ax1.set_xticks(range(len(x_labels_eval)))
    ax1.set_xticklabels(x_labels_eval, rotation=45, ha='right')
    ax1.legend(title='Evaluated Task')
    ax1.grid(True)
    
    x_labels_mmlu = df_mmlu['training_task'].tolist()
    if 'subject_avg_acc' in df_mmlu.columns:
        ax2.plot(df_mmlu['training_task'], df_mmlu['subject_avg_acc'], marker='s', linewidth=2.5, label='subject_avg_acc')
    for sub in subjects:
        ax2.plot(df_mmlu['training_task'], df_mmlu[sub], marker='^', label=sub)
    ax2.set_title('MMLU subject_acc Progression')
    ax2.set_xlabel('Training Progression (Assessed After Task)')
    ax2.set_ylabel('subject_acc')
    ax2.set_xticks(range(len(x_labels_mmlu)))
    ax2.set_xticklabels(x_labels_mmlu, rotation=45, ha='right')
    ax2.legend(title='MMLU Subject')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{run_base}_scores.png")
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    main()

