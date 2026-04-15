

def compute_rouge_metrics(dataset, preds, save_prefix=None):
    decoded_preds = skip_instructions(model, preds, tokenizer)
    references = [e["Instance"]["label"] for e in dataset]
    result = compute_metrics(predictions=decoded_preds, references=references)
    result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                              groups=dataset["Task"])
    result.update(result_per_task)
    categories = dataset["Dataset"]
    result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=categories)
    result.update(result_per_category)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    if save_prefix is not None:
        with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
            for example, pred in zip(dataset, decoded_preds):
                fout.write(json.dumps({
                    "Task": example["Task"],
                    "Dataset": example["Dataset"],
                    "Instance": example["Instance"],
                    "Prediction": pred
                }) + "\n")
    return result