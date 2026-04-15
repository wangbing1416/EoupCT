import json
import os
import time
import numpy as np
from transformers import Trainer, TrainingArguments, EvalPrediction, Seq2SeqTrainer, GenerationConfig
from transformers.trainer import *
from transformers.trainer_pt_utils import *
from typing import Dict, Optional
from typing import Sequence, Union, List, Tuple
import torch
import torch.distributed as dist
from evaluation.mmlu_runner import run_mmlu_eval, resolve_subjects
import logging

logger = logging.getLogger(__name__)


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    # If predictions_ids is a list of lists.
    if isinstance(predictions_ids, list):
        # First pad to the same length.
        max_len = max(len(x) for x in predictions_ids)
        padded = []
        for seq in predictions_ids:
            padded_seq = [token_id if token_id != ignore_idx else tokenizer.pad_token_id for token_id in seq]
            padded_seq = padded_seq + [tokenizer.pad_token_id] * (max_len - len(padded_seq))
            padded.append(padded_seq)
        predictions_ids = np.array(padded)
    else:
        # Already a numpy array; process directly.
        predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    ANSWER_PREFIX = "Output:"
    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    for pred in predictions:
        if tokenizer.bos_token:  # qwen3 has no tokenizer.bos_token
            pred = pred.replace(tokenizer.bos_token, '')
        pred = pred.replace(tokenizer.eos_token, '')
        pred = pred.replace(tokenizer.pad_token, '')
        if ANSWER_PREFIX in pred:
            splits = pred.split(ANSWER_PREFIX)
            final_predictions.append(splits[-1].strip())
        else:
            final_predictions.append(pred.strip())
    # final_predictions = predictions

    return final_predictions


class CLTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_task_name: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_task_name = eval_task_name  # Current evaluation task name.

    # Copied from Accelerate.
    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes.
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        # When extracting XLA graphs for compilation, max_size is 0,
        # so use an inequality check to avoid errors.
        if tensor.shape[1] >= max_size:
            return tensor

        # Then pad to the maximum size.
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.is_deepspeed_enabled:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None,  # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            # logger.info(f"max new token: {self._gen_kwargs.get("max_new_tokens")}")
            # logger.info(f"----->>>>>model.generation_config.max_length: {self.model.generation_config.max_length}")
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                # print(f"------>>>>>>>logits.shape: {logits.shape}")
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # gen_kwargs = self._gen_kwargs
        gen_kwargs = {
            "max_new_tokens": 50,
            "num_beams": 1,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
        }
        # gen_kwargs["synced_gpus"] = False

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if 'input_ids_wo_labels' in inputs:
            generation_inputs = inputs['input_ids_wo_labels']
        else:
            generation_inputs = inputs[self.model.main_input_name]

        if 'attention_mask_wo_labels' in inputs:
            attention_mask = inputs['attention_mask_wo_labels']

        elif "attention_mask" in inputs:
            attention_mask = inputs.get("attention_mask", None)
        else: attention_mask = None

        generation_config = GenerationConfig(**gen_kwargs)

        generated_tokens = self.model.generate(
            input_ids=generation_inputs,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        max_length = source_len + gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def evaluate_mmlu(
        self,
        mmlu_test_dir: str,
        mmlu_subjects: Optional[str] = None,
        ntrain: int = 5,
        max_length: int = 2048,
        batch_size: int = 1,
        verbose_debug: bool = False,
        debug_example_limit: int = 5,
    ) -> Optional[Dict[str, float]]:
        """
        Runs MMLU evaluation in a distributed manner using the trainer's model.
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Use unwrapped model for evaluation to avoid DDP/FSDP synchronization issues during inference loop
        # and to ensure we can control the forward pass (e.g. for simple generation/logits)
        model = self.accelerator.unwrap_model(self.model)

        # Ensure model is on the correct device
        # For FSDP/DS, the model might be on CPU or meta device until used, but unwrap_model usually handles gathering.
        # If standard DDP, model.module is on GPU. unwrap_model returns model.module.
        target_device = self.args.device

        # Fallback if unwrap_model returns CPU model but we have GPU
        # (Only move if not already on a cuda device)
        if target_device.type == "cuda" and next(model.parameters()).device.type == "cpu":
             model = model.to(target_device)

        # Resolve subjects
        if not os.path.exists(mmlu_test_dir):
            if self.is_world_process_zero():
                logger.warning(f"MMLU test dir not found: {mmlu_test_dir}")
            return None

        # Parse subjects from string or list
        if isinstance(mmlu_subjects, str):
            selected_subjects_list = [s.strip() for s in mmlu_subjects.split(",") if s.strip()]
        else:
            selected_subjects_list = mmlu_subjects

        # Get all available subjects first to ensure consistent ordering across ranks
        all_subjects = resolve_subjects(mmlu_test_dir, selected_subjects_list)

        if not all_subjects:
            return None

        # Distributed subject splitting
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        my_subjects = [s for i, s in enumerate(all_subjects) if i % world_size == rank]

        logger.info(f"[Rank {rank}] Evaluating {len(my_subjects)} MMLU subjects...")

        # Run evaluation on this rank's subset
        my_result = None
        if len(my_subjects) > 0:
            try:
                my_result = run_mmlu_eval(
                    model=model,
                    tokenizer=self.tokenizer,
                    mmlu_root_dir=mmlu_test_dir,
                    subjects=my_subjects,
                    ntrain=ntrain,
                    max_length=max_length,
                    batch_size=batch_size,
                    verbose_debug=verbose_debug and (rank == 0),
                    debug_example_limit=debug_example_limit,
                )
            except Exception as e:
                logger.error(f"[Rank {rank}] Error during MMLU evaluation: {e}")
                import traceback
                traceback.print_exc()

        # Prepare for gathering
        # Structure to gather: list of (subject, cors_list, count) or similar
        my_data = {
            "subject_cors": my_result["subject_cors"] if my_result else {},
            "subject_counts": my_result["subject_counts"] if my_result else {},
            "debug_examples": my_result["debug_examples"] if my_result else [],
        }

        # Gather results from all ranks
        if world_size > 1:
            all_data = [None for _ in range(world_size)]
            dist.all_gather_object(all_data, my_data)
        else:
            all_data = [my_data]

        # Aggregate on rank 0
        if self.is_world_process_zero():
            aggregated_data = {
                "subject_cors": {},
                "subject_counts": {},
                "debug_examples": []
            }

            for d in all_data:
                aggregated_data["subject_cors"].update(d["subject_cors"])
                aggregated_data["subject_counts"].update(d["subject_counts"])
                aggregated_data["debug_examples"].extend(d["debug_examples"])

            # Recalculate metrics
            from evaluation.evaluate_utils.categories_mmlu import categories, subcategories

            subject_acc = {}
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            all_cors = []

            for subject in all_subjects:
                if subject in aggregated_data["subject_cors"]:
                    cors = np.array(aggregated_data["subject_cors"][subject]) # reconstruct numpy array
                    if len(cors) == 0:
                         continue

                    subject_acc[subject] = float(np.mean(cors))
                    all_cors.append(cors)

                    for subcat in subcategories.get(subject, []):
                        subcat_cors[subcat].append(cors)
                        for cat, cat_subcats in categories.items():
                            if subcat in cat_subcats:
                                cat_cors[cat].append(cors)

            final_result = {
                "subjects": all_subjects,
                "subject_acc": subject_acc,
                "mmlu_subject_count": len(subject_acc),
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
                "debug_examples": aggregated_data["debug_examples"][:debug_example_limit], # Keep a few
            }

            return final_result

        return None
