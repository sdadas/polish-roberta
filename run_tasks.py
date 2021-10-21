import logging
import string
from random import choice

import fire
import fcntl
from datetime import datetime

from preprocess.processor import TaskProcessor
from train.evaluator import TaskEvaluatorBuilder
from tasks import *
from train.trainer import TaskTrainer
from train.writer import ModelWriter

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

    def prepare_task(self, resample: str):
        processor = TaskProcessor(self.task, self.input_dir, self.output_dir, self.model_dir, resample)
        processor.prepare()

    def train_task(self, train_epochs: int, fp16: bool, lr: str, max_sentences: int, update_freq: int,
                   ddp_backend: str, cpu_offload: bool, reinit_layers: int):
        train_size = self._count_train()
        trainer = TaskTrainer(self.task, self.output_dir, self.model_dir, train_size, lr=lr,
                              arch=self.arch, fp16=fp16, ddp_backend=ddp_backend, cpu_offload=cpu_offload)
        trainer.train(max_sentences, update_freq, train_epochs, reinit_layers, self.seed)

    def evaluate_task(self, verbose: bool=False, sharded_model: bool=False):
        builder = TaskEvaluatorBuilder(self.task, self.arch, self.model_dir, self.input_dir,
                                       output_dir=self.output_dir, verbose=verbose, sharded_model=sharded_model)
        evaluator = builder.build()
        return evaluator.evaluate(self.task_id)

    def save_model(self, output_dir: str):
        writer = ModelWriter(self.task, self.model_dir)
        writer.write_model(output_dir)

    def _count_train(self):
        train_path = os.path.join(self.output_dir, self.task.spec().output_path(), "train.input0")
        with open(train_path, "r", encoding="utf-8") as train_file:
            return len(train_file.readlines())

    def log_score(self, task_name: str, task_id: str, params: Dict, scores: Dict):
        now = datetime.now().strftime("%d/%m/%Y,%H:%M:%S")
        res = {"id": task_id, "task": task_name, "timestamp": now, "scores": scores, "params": params}
        with open("runlog.txt", "a", encoding="utf-8") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            json.dump(res, output_file)
            output_file.write("\n")
            fcntl.flock(output_file, fcntl.LOCK_UN)


def run_tasks(arch: str, model_dir: str, input_dir: str="data", output_dir: str="data_processed", lr: str="1e-5",
              tasks: str=None, train_epochs: int=10, fp16: bool=False, max_sentences: int=1, update_freq: int=16,
              evaluation_only: bool=False, resample: str=None, seed: int=None, verbose=False, ddp_backend: str=None,
              cpu_offload: bool=False, cv_folds: int=1, save_model_to: str=None, reinit_layers: int=None):
    assert arch in ("roberta_base", "roberta_large", "bart_base", "bart_large", "xlmr.xl")
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
        task = task_class()
        task_id = task_name.lower() + "_" + rand
        cross_validation = cv_folds > 1
        task_runs = CrossValidatedTask.cv_folds(task, cv_folds, seed) if cross_validation else [task]
        for idx, task_run in enumerate(task_runs):
            task_run_id = task_id
            if cross_validation: task_run_id += f"-fold{idx}"
            runner: TaskRunner = TaskRunner(task_run, task_run_id, input_dir, output_dir, model_dir, arch, seed)
            if not evaluation_only:
                runner.prepare_task(resample)
                runner.train_task(train_epochs, fp16, lr, max_sentences, update_freq, ddp_backend, cpu_offload, reinit_layers)
            sharded_model = ddp_backend == "fully_sharded" and cpu_offload
            if save_model_to is not None:
                runner.save_model(save_model_to)
            score = runner.evaluate_task(verbose, sharded_model)
            runner.log_score(task_name, task_run_id, params, score)


if __name__ == '__main__':
    fire.Fire(run_tasks)
