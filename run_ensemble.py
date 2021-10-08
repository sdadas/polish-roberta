import logging
from collections import Counter
from datetime import datetime
from random import choice
from typing import List, Tuple, Optional, Dict

import fire
import string
import fcntl

import json

from preprocess.processor import TaskProcessor
from tasks import TASKS, BaseTask
from train.evaluator import TaskEvaluatorBuilder, TaskEvaluator

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


class EnsemblePrediction(object):

    def __init__(self, y_true, task: BaseTask):
        self.y_true = y_true
        self.y_pred = []
        self.aggregate = self._vote_ensemble if task.spec().task_type == "classification" else self._avg_ensemble

    def add(self, val):
        self.y_pred.append(val)

    def _vote_ensemble(self):
        counter = Counter(self.y_pred)
        most_common: List[Tuple] = counter.most_common(1)
        return most_common[0][0]

    def _avg_ensemble(self):
        return sum(self.y_pred) / len(self.y_pred)


class EnsembleRunner(object):

    def __init__(self, arch: str, task_name: str, task: BaseTask, input_dir: str, output_dir: str):
        self.task_name = task_name
        self.task = task
        self.arch = arch
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.predictions: Optional[List[EnsemblePrediction]] = None
        self.evaluator = None

    def evaluate_model(self, model_dir: str):
        logging.info("generating predictions for model %s", model_dir)
        builder = TaskEvaluatorBuilder(self.task, self.arch, model_dir, pre_trained_model=True)
        self.evaluator: TaskEvaluator = builder.build()
        y_true, y_pred = self.evaluator.generate_predictions()
        if self.predictions is None:
            self.predictions = [EnsemblePrediction(val, self.task) for val in y_true]
        for idx, pred in enumerate(y_pred):
            self.predictions[idx].add(pred)

    def evaluate_ensemble(self, task_id: str):
        y_true = [val.y_true for val in self.predictions]
        y_pred = [val.aggregate() for val in self.predictions]
        return self.evaluator.evaluate_predictions(y_true, y_pred, task_id)

    def prepare_task(self, model_dir: str):
        processor = TaskProcessor(self.task, self.input_dir, self.output_dir, model_dir)
        processor.prepare()

    def log_score(self, task_name: str, task_id: str, params: Dict, scores: Dict):
        now = datetime.now().strftime("%d/%m/%Y,%H:%M:%S")
        res = {"id": task_id, "task": task_name, "timestamp": now, "scores": scores, "params": params, "ensemble": True}
        with open("runlog.txt", "a", encoding="utf-8") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            json.dump(res, output_file)
            output_file.write("\n")
            fcntl.flock(output_file, fcntl.LOCK_UN)


def run_ensemble(arch: str, task: str, model_dirs: List[str], input_dir: str="data", output_dir: str="data_processed"):
    params = dict(locals())
    task_name = task
    task_class = TASKS.get(task)
    if task_class is None: raise Exception(f"Unknown task {task}")
    task = task_class()
    runner = EnsembleRunner(arch, task_name, task, input_dir, output_dir)
    for model_dir in model_dirs:
        runner.prepare_task(model_dir)
        runner.evaluate_model(model_dir)
    rand = ''.join(choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
    task_id = task_name.lower() + "_" + rand
    scores = runner.evaluate_ensemble(task_id)
    runner.log_score(task_name, task_id, params, scores)


if __name__ == '__main__':
    fire.Fire(run_ensemble)
