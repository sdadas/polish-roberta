import logging
import os
from typing import List

from fairseq.models.roberta import RobertaHubInterface
from tasks import BaseTask


class TaskEvaluator(object):

    def __init__(self, task: BaseTask, task_id: str, model: RobertaHubInterface, data_path: str, output_dir: str):
        self.task: BaseTask = task
        self.task_id: str = task_id
        self.model: RobertaHubInterface = model
        self.data_path: str = data_path
        self.output_dir = output_dir
        self.model.cuda()
        self.model.eval()

    def _get_label(self, label):
        return self.model.task.label_dictionary.string([label + self.model.task.label_dictionary.nspecial])

    def evaluate(self):
        y_true = []
        y_pred = []
        logits = False
        logging.info("evaluating task %s", self.task.spec().output_dir)
        if self.task.spec().task_type == "classification":
            get_pred = lambda v: self._get_label(v.argmax().item())
            get_true = lambda v: v.label
        else:
            get_pred = lambda v: v.item()
            get_true = lambda v: float(v.label)
            logits = True
        for record in self.task.read(self.data_path, "test"):
            tokens = self.model.encode(*record.inputs)
            if tokens.size()[0] > 512:
                tokens = tokens[0:512]
            prediction = self.model.predict("sentence_classification_head", tokens, return_logits=logits)
            y_true.append(get_true(record))
            y_pred.append(get_pred(prediction))
        scores = self.task.spec().evaluation_metric(y_true, y_pred)
        logging.info("scores = %s", scores.__repr__())
        self.save_results(y_pred)
        return scores

    def save_results(self, y_pred: List[any]):
        output_path = os.path.join(self.output_dir, f"{self.task_id}.txt")
        sample_value = y_pred[0]
        output_func = lambda v: "%.5f" % (v,) if isinstance(sample_value, float) else str
        if hasattr(self.task, "format_output"):
            output_func = self.task.format_output
        with open(output_path, "w", encoding="utf-8") as output_file:
            for value in y_pred:
                output_file.write(output_func(value))
                output_file.write("\n")
