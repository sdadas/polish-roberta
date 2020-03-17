import logging
from fairseq.models.roberta import RobertaHubInterface
from tasks import BaseTask


class TaskEvaluator(object):

    def __init__(self, task: BaseTask, model: RobertaHubInterface, data_path: str):
        self.task: BaseTask = task
        self.model: RobertaHubInterface = model
        self.data_path: str = data_path
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
            prediction = self.model.predict("sentence_classification_head", tokens, return_logits=logits)
            if prediction.size()[0] > 512:
                prediction = prediction[0:512]
            y_true.append(get_true(record))
            y_pred.append(get_pred(prediction))
        score = self.task.spec().evaluation_metric(y_true, y_pred)
        logging.info("score = %.6f", score)
        return score
