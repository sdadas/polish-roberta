import argparse
import logging
import os
from typing import List, Callable

import torch.cuda
from fairseq import utils
from fairseq.data.data_utils import collate_tokens
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from tasks import BaseTask, DataExample
from train.bart import CustomBARTHubInterface


class TaskEvaluator(object):

    def __init__(self, task: BaseTask, model: RobertaHubInterface, data_path: str, output_dir: str, verbose=False):
        self.task: BaseTask = task
        self.model: RobertaHubInterface = model
        self.data_path: str = data_path
        self.output_dir = output_dir
        if hasattr(model, "cfg"):
            try: self.maxlen = model.cfg.task.max_positions
            except: self.maxlen = model.cfg.task.tokens_per_sample
        else:
            self.maxlen = model.args.max_positions
        self.log_predictions = verbose
        self.model.cuda()
        self.model.eval()
        self._init_prediction_settings()

    def _init_prediction_settings(self):
        self.logits = False
        if self.task.spec().task_type == "classification":
            self.get_pred = lambda v: self._get_label(v.argmax().item())
            self.get_true = lambda v: v.label
        else:
            self.get_pred = lambda v: v.item()
            self.get_true = lambda v: float(v.label)
            self.logits = True
        if hasattr(self.task, "postprocess_prediction"):
            get_pred_original = self.get_pred
            postprocess: Callable = self.task.postprocess_prediction
            self.get_pred = lambda v: postprocess(get_pred_original(v))

    def _get_label(self, label):
        return self.model.task.label_dictionary.string([label + self.model.task.label_dictionary.nspecial])

    def generate_predictions(self):
        y_true = []
        y_pred = []
        logging.info("generating predictions for task %s", self.task.spec().output_dir)
        for record in self.task.read(self.data_path, "test"):
            y_true.append(self.get_true(record) if record.label is not None else None)
            y_pred.append(self.predict(record))
        return y_true, y_pred

    def evaluate_predictions(self, y_true, y_pred, task_id: str="sample_task"):
        if y_true[0] is None:
            logging.info("No test labels available, skipping evaluation for task %s", self.task.spec().output_dir)
            scores = {}
        else:
            scores = self.task.spec().evaluation_metric(y_true, y_pred)
            logging.info("scores = %s", scores.__repr__())
        self.save_results(y_pred, task_id)
        return scores

    def evaluate(self, task_id: str="sample_task"):
        y_true, y_pred = self.generate_predictions()
        return self.evaluate_predictions(y_true, y_pred, task_id)

    def predict(self, record: DataExample, logits: bool=False):
        tokens = self.encode(record)
        prediction = self.model.predict("sentence_classification_head", tokens, return_logits=logits or self.logits)
        if self.log_predictions and record.label is not None: self.log_prediction(prediction, record)
        if logits: return prediction
        else: return self.get_pred(prediction)

    def predict_batch(self, records: List[DataExample]):
        encoded = [self.encode(record) for record in records]
        batch = collate_tokens(encoded, pad_idx=1)
        predictions = self.model.predict("sentence_classification_head", batch)
        return [self.get_pred(pred) for _, pred in enumerate(predictions)]

    def encode(self, record: DataExample):
        tokens = self.model.encode(*record.inputs)
        if tokens.size()[0] > self.maxlen:
            tokens = tokens[0:self.maxlen]
        return tokens

    def log_prediction(self, prediction, record: DataExample):
        y_pred = self.get_pred(prediction)
        y_true = record.label
        x = "\t".join(record.inputs)
        logging.info(f"true={y_true}\tpred={y_pred}\t{x}")

    def save_results(self, y_pred: List[any], task_id: str):
        output_path = os.path.join(self.output_dir, f"{task_id}.txt")
        sample_value = y_pred[0]
        output_func = (lambda v: "%.5f" % (v,)) if isinstance(sample_value, float) else str
        if hasattr(self.task, "format_output"):
            output_func = self.task.format_output
        with open(output_path, "w", encoding="utf-8") as output_file:
            for value in y_pred:
                output_file.write(output_func(value))
                output_file.write("\n")


class TaskEvaluatorBuilder(object):

    def __init__(self, task: BaseTask, arch: str, model_dir: str, input_dir: str="data", pre_trained_model=False,
                 output_dir: str="data_processed", verbose=False, sharded_model=False):
        self.task = task
        self.arch = arch
        self.model_dir = model_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.verbose = verbose
        self.model_name = os.path.basename(model_dir)
        self.pre_trained_model = pre_trained_model
        self.task_output_dir: str = os.path.join(self.output_dir, f"{task.spec().output_path()}-bin")
        self.sharded_model = sharded_model

    def build(self) -> TaskEvaluator:
        checkpoints_output_dir = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        checkpoint_file = "checkpoint_last.pt" if self.task.spec().no_dev_set else "checkpoint_best.pt"
        model_classes = {"roberta": (RobertaModel, RobertaHubInterface), "bart": (BARTModel, CustomBARTHubInterface)}
        arch_type = self.arch.split("_")[0]
        if arch_type.startswith("xlmr"): arch_type = "roberta"
        model_class = model_classes[arch_type][0]
        spm_path = os.path.join(self.model_dir, "sentencepiece.bpe.model")
        if self.pre_trained_model:
            checkpoints_output_dir = self.model_dir
            checkpoint_file = "model.pt"
        loaded = self.from_pretrained(
            model_name_or_path=checkpoints_output_dir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=self.task_output_dir,
            bpe="sentencepiece",
            sentencepiece_model=spm_path,
            sentencepiece_vocab=spm_path,
            load_checkpoint_heads=True,
            archive_map=model_class.hub_models(),
            num_shards=torch.cuda.device_count() if self.sharded_model else 1,
            strict=not self.sharded_model
        )
        model_interface = model_classes[arch_type][1]
        if isinstance(loaded, model_interface): model = loaded
        else: model = model_interface(loaded['args'], loaded['task'], loaded['models'][0])
        evaluator = TaskEvaluator(self.task, model, self.input_dir, checkpoints_output_dir, self.verbose)
        return evaluator

    def from_pretrained(self, model_name_or_path, checkpoint_file="model.pt", data_name_or_path=".",
                        archive_map=None, num_shards=1, strict=False, **kwargs):
        from fairseq import checkpoint_utils, file_utils

        if archive_map is not None:
            if model_name_or_path in archive_map:
                model_name_or_path = archive_map[model_name_or_path]
            if data_name_or_path is not None and data_name_or_path in archive_map:
                data_name_or_path = archive_map[data_name_or_path]

            # allow archive_map to set default arg_overrides (e.g., tokenizer, bpe)
            # for each model
            if isinstance(model_name_or_path, dict):
                for k, v in model_name_or_path.items():
                    if k == "checkpoint_file":
                        checkpoint_file = v
                    elif (
                            k != "path"
                            # only set kwargs that don't already have overrides
                            and k not in kwargs
                    ):
                        kwargs[k] = v
                model_name_or_path = model_name_or_path["path"]

        model_path = file_utils.load_archive_file(model_name_or_path)

        # convenience hack for loading data and BPE codes from model archive
        if data_name_or_path.startswith("."):
            kwargs["data"] = os.path.abspath(os.path.join(model_path, data_name_or_path))
        else:
            kwargs["data"] = file_utils.load_archive_file(data_name_or_path)
        for file, arg in {
            "code": "bpe_codes",
            "bpecodes": "bpe_codes",
            "sentencepiece.bpe.model": "sentencepiece_model",
        }.items():
            path = os.path.join(model_path, file)
            if os.path.exists(path):
                kwargs[arg] = path

        if "user_dir" in kwargs:
            utils.import_user_module(argparse.Namespace(user_dir=kwargs["user_dir"]))

        models, args, task = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(os.pathsep)],
            arg_overrides=kwargs, num_shards=num_shards, strict=strict
        )
        return {"args": args, "task": task, "models": models}