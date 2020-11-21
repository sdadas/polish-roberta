import logging
import os
from typing import List, Callable

from fairseq import hub_utils
from fairseq.models.bart import BARTModel
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from tasks import BaseTask, DataExample
from train.bart import CustomBARTHubInterface


class TaskEvaluator(object):

    def __init__(self, task: BaseTask, model: RobertaHubInterface, data_path: str, output_dir: str):
        self.task: BaseTask = task
        self.model: RobertaHubInterface = model
        self.data_path: str = data_path
        self.output_dir = output_dir
        self.maxlen = model.args.max_positions
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

    def evaluate(self, task_id: str="sample_task"):
        y_true = []
        y_pred = []
        logging.info("generating predictions for task %s", self.task.spec().output_dir)
        for record in self.task.read(self.data_path, "test"):
            y_true.append(self.get_true(record) if record.label is not None else None)
            y_pred.append(self.predict(record))
        if y_true[0] is None:
            logging.info("No test labels available, skipping evaluation for task %s", self.task.spec().output_dir)
            scores = {}
        else:
            scores = self.task.spec().evaluation_metric(y_true, y_pred)
            logging.info("scores = %s", scores.__repr__())
        self.save_results(y_pred, task_id)
        return scores

    def predict(self, record: DataExample, logits: bool=False):
        tokens = self.model.encode(*record.inputs)
        if tokens.size()[0] > self.maxlen:
            tokens = tokens[0:self.maxlen]
        prediction = self.model.predict("sentence_classification_head", tokens, return_logits=logits or self.logits)
        if logits: return prediction
        else: return self.get_pred(prediction)

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

    def __init__(self, task: BaseTask, arch: str, model_dir: str, input_dir: str="data", output_dir: str="data_processed"):
        self.task = task
        self.arch = arch
        self.model_dir = model_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = os.path.basename(model_dir)
        self.task_output_dir: str = os.path.join(self.output_dir, f"{task.spec().output_path()}-bin")

    def build(self) -> TaskEvaluator:
        checkpoints_output_dir = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        checkpoint_file = "checkpoint_last.pt" if self.task.spec().no_dev_set else "checkpoint_best.pt"
        model_classes = {"roberta": (RobertaModel, RobertaHubInterface), "bart": (BARTModel, CustomBARTHubInterface)}
        arch_type = self.arch.split("_")[0]
        model_class = model_classes[arch_type][0]
        spm_path = os.path.join(self.model_dir, "sentencepiece.bpe.model")
        loaded = hub_utils.from_pretrained(
            model_name_or_path=checkpoints_output_dir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=self.task_output_dir,
            bpe="sentencepiece",
            sentencepiece_model=spm_path,
            sentencepiece_vocab=spm_path,
            load_checkpoint_heads=True,
            archive_map=model_class.hub_models()
        )
        model_interface = model_classes[arch_type][1]
        model = model_interface(loaded['args'], loaded['task'], loaded['models'][0])
        evaluator = TaskEvaluator(self.task, model, self.input_dir, checkpoints_output_dir)
        return evaluator

