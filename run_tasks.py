import string
from random import choice

import fire
import fcntl
from datetime import datetime
import json

from preprocess.processor import TaskProcessor
from train.evaluator import TaskEvaluatorBuilder
from tasks import *
from train.trainer import TaskTrainer

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


class TaskRunner(object):

    def __init__(self, task: BaseTask, task_id: str, input_dir: str, output_dir: str, model_dir: str, arch: str, seed: int):
        self.task: BaseTask = task
        self.task_id: str = task_id
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.model_dir: str = model_dir
        self.seed = seed
        self.arch = arch
        self.model_name: str = os.path.basename(model_dir)
        self.task_output_dir: str = os.path.join(self.output_dir, f"{task.spec().output_path()}-bin")

    def prepare_task(self, resample: str, token_shapes: bool):
        processor = TaskProcessor(self.task, self.input_dir, self.output_dir, self.model_dir, resample, token_shapes)
        processor.prepare()

    def train_task(self, train_epochs: int, fp16: bool, max_sentences: int, update_freq: int, token_shapes: bool):
        train_size = self._count_train()
        trainer = TaskTrainer(self.task, self.output_dir, self.model_dir, train_size,
                              arch=self.arch, fp16=fp16, token_shapes=token_shapes)
        trainer.train(train_epochs=train_epochs, max_sentences=max_sentences, update_freq=update_freq)

    def evaluate_task(self):
        builder = TaskEvaluatorBuilder(self.task, self.arch, self.model_dir, self.input_dir, self.output_dir)
        evaluator = builder.build()
        return evaluator.evaluate(self.task_id)

    def _count_train(self):
        return sum(1 for _ in self.task.read(self.input_dir, "train"))

    def log_score(self, task_name: str, task_id: str, params: Dict, scores: Dict):
        now = datetime.now().strftime("%d/%m/%Y,%H:%M:%S")
        res = {"id": task_id, "task": task_name, "timestamp": now, "scores": scores, "params": params}
        with open("runlog.txt", "a", encoding="utf-8") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            json.dump(res, output_file)
            output_file.write("\n")
            fcntl.flock(output_file, fcntl.LOCK_UN)


def run_tasks(arch: str, model_dir: str, input_dir: str="data", output_dir: str="data_processed",
              tasks: str=None, train_epochs: int=10, fp16: bool=False, max_sentences: int=1, update_freq: int=16,
              evaluation_only: bool=False, resample: str=None, token_shapes: bool=False, seed: int=None):
    assert arch in ("roberta_base", "roberta_large", "bart_base", "bart_large")
    params = locals()
    if tasks is None:
        task_names = [key for key in TASKS.keys()]
        task_classes = [val for val in TASKS.values()]
    else:
        task_names = [task_name.strip() for task_name in tasks.split(",")]
        task_classes = [TASKS.get(task_name) for task_name in task_names]
    logging.info("Running training and evaluation for tasks %s", task_names.__repr__())
    for idx, task_class in enumerate(task_classes):
        task_name = task_names[idx]
        if task_class is None: raise Exception(f"Unknown task {task_name}")
        rand = ''.join(choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
        task_id = task_name.lower() + "_" + rand
        task = task_class()
        runner: TaskRunner = TaskRunner(task, task_id, input_dir, output_dir, model_dir, arch, seed)
        if not evaluation_only:
            runner.prepare_task(resample, token_shapes)
            runner.train_task(train_epochs, fp16, max_sentences, update_freq, token_shapes)
        score = runner.evaluate_task()
        runner.log_score(task_name, task_id, params, score)


if __name__ == '__main__':
    fire.Fire(run_tasks)
