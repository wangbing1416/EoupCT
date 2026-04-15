# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# from evaluation.evaluate_utils import categories_mmlu
from evaluation.evaluate_utils.categories_mmlu import categories, subcategories
from typing import List, Tuple, Union

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(shot, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        # k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, shot)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device_)

        while input_ids.shape[-1] > 2048:
            shot -= 1
            train_prompt = gen_prompt(dev_df, subject, shot)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device_)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = generate(model, tokenizer, prompt)
        logits = logits[0, -1]
        # logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer.encode("A")[-1]],
                        logits[tokenizer.encode("B")[-1]],
                        logits[tokenizer.encode("C")[-1]],
                        logits[tokenizer.encode("D")[-1]],
                    ]
                ).float(), dim=0,
            ).detach().cpu().numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def evaluate(model, tokenizer, data_dir, save_dir='results', shot: int = 5):

    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model.name_or_path_.split("/")[-1]))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model.name_or_path_.split("/")[-1])))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: shot]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(shot, subject, model, tokenizer, dev_df, test_df)  # evaluate function **
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model.name_or_path_.split("/")[-1])] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model.name_or_path_.split("/")[-1], choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(model.name_or_path_.split("/")[-1]), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        save_dir, "accuracies_{}.json".format(model.name_or_path_.split("/")[-1])
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))

    return weighted_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--bit", "-b", type=int, default=8)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="<MODEL_DIR>")
    args = parser.parse_args()
    # evaluate(args)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    evaluate(model, tokenizer, args.data_dir, args.save_dir, args.ntrain)
# python3 evaluate_hf.py -m <MODEL_DIR>
