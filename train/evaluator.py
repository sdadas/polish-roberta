import logging
import os
from typing import List, Callable

from fairseq.models.roberta import RobertaHubInterface
from tasks import BaseTask, DataExample


class TaskEvaluator(object):

    def __init__(self, task: BaseTask, task_id: str, model: RobertaHubInterface, data_path: str, output_dir: str):
        self.task: BaseTask = task
        self.task_id: str = task_id
        self.model: RobertaHubInterface = model
        self.data_path: str = data_path
        self.output_dir = output_dir
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

    def evaluate(self):
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
        self.save_results(y_pred)
        return scores

    def predict(self, record: DataExample, logits: bool=False):
        tokens = self.model.encode(*record.inputs)
        if tokens.size()[0] > 512:
            tokens = tokens[0:512]
        prediction = self.model.predict("sentence_classification_head", tokens, return_logits=logits or self.logits)
        if logits: return prediction
        else: return self.get_pred(prediction)

    def save_results(self, y_pred: List[any]):
        output_path = os.path.join(self.output_dir, f"{self.task_id}.txt")
        sample_value = y_pred[0]
        output_func = (lambda v: "%.5f" % (v,)) if isinstance(sample_value, float) else str
        if hasattr(self.task, "format_output"):
            output_func = self.task.format_output
        with open(output_path, "w", encoding="utf-8") as output_file:
            for value in y_pred:
                output_file.write(output_func(value))
                output_file.write("\n")
