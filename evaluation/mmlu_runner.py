import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluation.evaluate_utils.categories_mmlu import categories, subcategories


_CHOICES = ["A", "B", "C", "D"]


def _format_subject(subject: str) -> str:
    s = ""
    for entry in subject.split("_"):
        s += " " + entry
    return s


def _format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(_CHOICES[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def _gen_prompt(dev_df: pd.DataFrame, subject: str, k: int = -1) -> str:
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        _format_subject(subject)
    )
    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += _format_example(dev_df, i)
    return prompt


def _get_choice_token_ids(tokenizer):
    # Match the legacy script as closely as possible: use the raw encode("A")[-1] style.
    return [tokenizer.encode(choice)[-1] for choice in _CHOICES]


def _resolve_mmlu_subject_paths(mmlu_root_dir: str, subject: str):
    # Legacy layout is data_dir/dev/*.csv and data_dir/test/*.csv.
    nested_dev = os.path.join(mmlu_root_dir, "dev", subject + "_dev.csv")
    nested_test = os.path.join(mmlu_root_dir, "test", subject + "_test.csv")
    if os.path.exists(nested_dev) and os.path.exists(nested_test):
        return nested_dev, nested_test
    return None, None


def resolve_subjects(mmlu_test_dir: str, selected_subjects: Optional[List[str]] = None) -> List[str]:
    candidate_dir = os.path.join(mmlu_test_dir, "test")
    if not os.path.isdir(candidate_dir):
        return []
    all_subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(candidate_dir) if f.endswith("_test.csv")]
    )
    if not selected_subjects:
        return all_subjects
    available_set = set(all_subjects)
    return [s for s in selected_subjects if s in available_set]


def _get_model_dtype_and_device(model):
    try:
        p = next(model.parameters())
        return p.dtype, p.device
    except StopIteration:
        return None, getattr(model, "device", None)


def _print_model_dtype_summary(model):
    counts = {}
    for name, param in model.named_parameters():
        key = str(param.dtype)
        counts[key] = counts.get(key, 0) + 1
    print(f"[MMLU-MODEL-SUMMARY] param_dtype_counts={counts}")
    for name, param in list(model.named_parameters())[:20]:
        print(f"[MMLU-MODEL-PARAM] {name} dtype={param.dtype} device={param.device} shape={tuple(param.shape)}")


def _normalize_model_dtypes(model, target_dtype):
    if target_dtype not in (torch.float16, torch.bfloat16):
        return 0
    changed = 0
    for name, param in model.named_parameters():
        if param.dtype in (torch.float16, torch.bfloat16) and param.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)
            changed += 1
    return changed


def run_mmlu_eval(
    model,
    tokenizer,
    mmlu_root_dir: str,
    subjects: Optional[List[str]] = None,
    ntrain: int = 5,
    max_length: int = 2048,
    batch_size: int = 16,
    verbose_debug: bool = False,
    debug_example_limit: int = 5,
) -> Dict[str, object]:
    selected_subjects = resolve_subjects(mmlu_root_dir, subjects)
    choice_token_ids = _get_choice_token_ids(tokenizer)

    model_param_dtype, model_param_device = _get_model_dtype_and_device(model)
    if verbose_debug:
        print(f"[MMLU-DTYPE] model_param_dtype={model_param_dtype} model_param_device={model_param_device} model_device={getattr(model, 'device', None)}")
        _print_model_dtype_summary(model)

    normalized = _normalize_model_dtypes(model, model_param_dtype)
    if verbose_debug:
        print(f"[MMLU-DTYPE] normalized_params_to={model_param_dtype} changed_params={normalized}")

    subject_acc = {}
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}
    all_cors = []
    debug_examples = []
    total_loaded_test_samples = 0
    total_loaded_dev_samples = 0

    if verbose_debug:
        print(f"[MMLU-LOAD] resolved_subject_count={len(selected_subjects)} root_dir={mmlu_root_dir}")

    for subject in selected_subjects:
        dev_path, test_path = _resolve_mmlu_subject_paths(mmlu_root_dir, subject)
        if dev_path is None or test_path is None:
            if verbose_debug:
                print(f"[MMLU-WARN] missing files for subject={subject} under root={mmlu_root_dir}")
            continue

        dev_df = pd.read_csv(dev_path, header=None)[:ntrain]
        test_df = pd.read_csv(test_path, header=None)
        total_loaded_dev_samples += len(dev_df)
        total_loaded_test_samples += len(test_df)

        if verbose_debug:
            print(
                f"[MMLU-LOAD] subject={subject} dev_samples={len(dev_df)} test_samples={len(test_df)} dev_path={dev_path} test_path={test_path}"
            )

        cors = []
        all_probs = []
        model.eval()

        for i in tqdm(range(test_df.shape[0]), desc=f"mmlu:{subject}", leave=False):
            k = ntrain
            prompt_end = _format_example(test_df, i, include_answer=False)
            train_prompt = _gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            while input_ids.shape[-1] > max_length and k > 0:
                k -= 1
                train_prompt = _gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            label = test_df.iloc[i, test_df.shape[1] - 1]

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1]

            probs = torch.nn.functional.softmax(
                torch.tensor([logits[tokenizer.encode(choice)[-1]] for choice in _CHOICES], dtype=torch.float32),
                dim=0,
            ).detach().cpu().numpy()
            pred = _CHOICES[int(np.argmax(probs))]
            cor = pred == label
            cors.append(cor)
            all_probs.append(probs)

            if verbose_debug and len(debug_examples) < debug_example_limit:
                debug_examples.append(
                    {
                        "subject": subject,
                        "question": str(test_df.iloc[i, 0])[:200],
                        "label": str(label),
                        "pred": pred,
                        "probs": {
                            "A": float(probs[0]),
                            "B": float(probs[1]),
                            "C": float(probs[2]),
                            "D": float(probs[3]),
                        },
                    }
                )

        cors = np.array(cors)
        all_probs = np.array(all_probs)
        acc = float(np.mean(cors)) if len(cors) > 0 else 0.0
        subject_acc[subject] = acc
        all_cors.append(cors)

        if verbose_debug:
            print(f"Average accuracy {acc:.3f} - {subject}")

        for subcat in subcategories.get(subject, []):
            subcat_cors[subcat].append(cors)
            for cat, cat_subcats in categories.items():
                if subcat in cat_subcats:
                    cat_cors[cat].append(cors)

    result = {
        "subjects": selected_subjects,
        "subject_acc": subject_acc,
        "subject_cors": {s: c.tolist() for s, c in zip(selected_subjects, all_cors)},
        "subject_counts": {s: len(c) for s, c in zip(selected_subjects, all_cors)},
        "mmlu_subject_count": len(selected_subjects),
        "mmlu_weighted_acc": float(np.mean(np.concatenate(all_cors))) if all_cors else 0.0,
        "category_acc": {
            cat: float(np.mean(np.concatenate(cat_cors[cat])))
            for cat in cat_cors
            if len(cat_cors[cat]) > 0
        },
        "subcategory_acc": {
            subcat: float(np.mean(np.concatenate(subcat_cors[subcat])))
            for subcat in subcat_cors
            if len(subcat_cors[subcat]) > 0
        },
        "debug_examples": debug_examples,
    }

    if verbose_debug:
        for subcat in subcat_cors:
            if len(subcat_cors[subcat]) > 0:
                print(f"Average accuracy {float(np.mean(np.concatenate(subcat_cors[subcat]))):.3f} - {subcat}")
        for cat in cat_cors:
            if len(cat_cors[cat]) > 0:
                print(f"Average accuracy {float(np.mean(np.concatenate(cat_cors[cat]))):.3f} - {cat}")
        print(f"Average accuracy: {result['mmlu_weighted_acc']:.3f}")
        if debug_examples:
            print("MMLU representative examples:")
            for idx, ex in enumerate(debug_examples[:debug_example_limit], start=1):
                print(
                    f"[MMLU-EX-{idx}] subject={ex['subject']} label={ex['label']} pred={ex['pred']} probs={ex['probs']} question={ex['question']}"
                )
        print(
            f"[MMLU-LOAD] total_subjects={len(selected_subjects)} total_dev_samples={total_loaded_dev_samples} total_test_samples={total_loaded_test_samples}"
        )

    return result

