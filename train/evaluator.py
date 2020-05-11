import logging
import os
import numpy as np
from typing import List, Callable

# from fairseq.models.roberta import RobertaHubInterface
from tasks import BaseTask


class TaskEvaluator(object):

    def __init__(self, task: BaseTask, task_id: str, model, data_path: str, output_dir: str, model_type: str):
        self.task: BaseTask = task
        self.task_id: str = task_id
        self.model = model
        self.data_path: str = data_path
        self.output_dir = output_dir
        self.model_type = model_type
        if model_type == 'fairseq':
            self.model.cuda()
            self.model.eval()

    def _get_label(self, label):
        return self.model.task.label_dictionary.string([label + self.model.task.label_dictionary.nspecial])

    def evaluate(self):
        y_true = []
        y_pred = []
        logits = False
        logging.info("generating predictions for task %s", self.task.spec().output_dir)
        if self.task.spec().task_type == "classification":
            get_pred = lambda v: self._get_label(v.argmax().item())
            get_true = lambda v: v.label
        else:
            get_pred = lambda v: v.item()
            get_true = lambda v: float(v.label)
            logits = True
        if hasattr(self.task, "postprocess_prediction"):
            get_pred_original = get_pred
            postprocess: Callable = self.task.postprocess_prediction
            get_pred = lambda v: postprocess(get_pred_original(v))

        if self.model_type == 'fairseq':
            for record, prediction in self._fairseq_predict(logits):
                y_true.append(get_true(record))
                y_pred.append(get_pred(prediction))
        elif self.model_type == 'transformers':
            y_true, y_pred = self._transformers_predict()

        if y_true[0] is None:
            logging.info("No test labels available, skipping evaluation for task %s", self.task.spec().output_dir)
            scores = {}
        else:
            scores = self.task.spec().evaluation_metric(y_true, y_pred)
            logging.info("scores = %s", scores.__repr__())
        self.save_results(y_pred)
        return scores

    def _fairseq_predict(self, logits):
        for record in self.task.read(self.data_path, "test"):
            tokens = self.model.encode(*record.inputs)
            if tokens.size()[0] > 512:
                tokens = tokens[0:512]
            prediction = self.model.predict("sentence_classification_head", tokens, return_logits=logits)
            yield record if record.label is not None else None, prediction

    def _transformers_predict(self):
        test_data = self.task.read_csv(self.data_path, 'test', label_first=False, normalize=True)
        _, model_outputs, _ = self.model.eval_model(test_data)
        if self.task.spec().task_type == 'regression':
            y_pred = model_outputs
        else:
            y_pred = np.argmax(model_outputs, axis=1)
        y_true = test_data['labels'].tolist()
        return y_true, y_pred

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
