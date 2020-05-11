import string
from random import choice

import fire
# from fairseq import hub_utils
# from fairseq.models.roberta import RobertaModel, RobertaHubInterface
# import fcntl
import torch

from simpletransformers.classification import ClassificationModel

from preprocess.processor import TaskProcessor
from train.evaluator import TaskEvaluator
from tasks import *
from train.trainer import TaskTrainer

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


TASKS = {
    "WCCRS_HOTELS":    WCCRSHotelsTask,
    "WCCRS_MEDICINE":  WCCRSMedicineTask,
    "SICK-E":          SICKEntailmentTask,
    "SICK-R":          SICKRelatednessTask,
    "8TAGS":           EightTagsTask,
    "KLEJ-NKJP":       KLEJNKJPTask,
    "KLEJ-CDS-E":      KLEJCDSEntailmentTask,
    "KLEJ-CDS-R":      KLEJCDSRelatednessTask,
    "KLEJ-CBD":        KLEJCBDTask,
    "KLEJ-POLEMO-IN":  KLEJPolEmoINTask,
    "KLEJ-POLEMO-OUT": KLEJPolEmoOUTTask,
    "KLEJ-DYK":        KLEJDYKTask,
    "KLEJ-PSC":        KLEJPSCTask,
    "KLEJ-ECR":        KLEJECRRegressionTask
}


class TaskRunner(object):

    def __init__(self, task: BaseTask, task_id: str, input_dir: str, output_dir: str, model_dir: str, seed: int, model_type: str, arch: str = None):
        self.task: BaseTask = task
        self.task_id: str = task_id
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.model_dir: str = model_dir
        self.seed = seed
        self.model_name: str = os.path.basename(model_dir)
        self.model_type: str = model_type
        self.task_output_dir: str = os.path.join(self.output_dir, f"{task.spec().output_path()}-bin")
        self.arch = arch

    def prepare_task(self, resample: str):
        processor = TaskProcessor(self.task, self.input_dir, self.output_dir, self.model_dir, resample,
                                  model_format=self.model_type)
        processor.prepare()

    def train_task(self, arch: str, train_epochs: int, fp16: bool, max_sentences: int, update_freq: int):
        train_size = self._count_train()
        trainer = TaskTrainer(self.task, self.output_dir, self.model_dir, train_size, arch=arch, fp16=fp16,
                              model_type=self.model_type)
        trainer.train(train_epochs=train_epochs, max_sentences=max_sentences, update_freq=update_freq)

    def evaluate_task(self):
        if self.model_type == 'transformers':
            checkpoints_output_dir = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
            if not self.task.spec().no_dev_set:
                checkpoints_output_dir = os.path.join(checkpoints_output_dir, 'best_checkpoint')
            model = ClassificationModel(self.arch, checkpoints_output_dir, use_cuda=torch.cuda.is_available(),
                                        args={'regression': self.task.spec().task_type == 'regression'})

            evaluator = TaskEvaluator(self.task, self.task_id, model, self.output_dir, checkpoints_output_dir, model_type='transformers')
            return evaluator.evaluate()
        elif self.model_type == 'fairseq':
            pass
            # checkpoints_output_dir = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
            # checkpoint_file = "checkpoint_last.pt" if self.task.spec().no_dev_set else "checkpoint_best.pt"
            # loaded = hub_utils.from_pretrained(
            #     model_name_or_path=checkpoints_output_dir,
            #     checkpoint_file=checkpoint_file,
            #     data_name_or_path=self.task_output_dir,
            #     bpe="sentencepiece",
            #     sentencepiece_vocab=os.path.join(self.model_dir, "sentencepiece.bpe.model"),
            #     load_checkpoint_heads=True,
            #     archive_map=RobertaModel.hub_models()
            # )
            # roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
            # evaluator = TaskEvaluator(self.task, self.task_id, roberta, self.input_dir, checkpoints_output_dir)
            # return evaluator.evaluate()

    def _count_train(self):
        return sum(1 for _ in self.task.read(self.input_dir, "train"))

    def log_score(self, task_name: str, task_id: str, params: Dict, scores: Dict):
        pass
        # now = datetime.now().strftime("%d/%m/%Y,%H:%M:%S")
        # res = {"id": task_id, "task": task_name, "timestamp": now, "scores": scores, "params": params}
        # with open("runlog.txt", "a", encoding="utf-8") as output_file:
        #     fcntl.flock(output_file, fcntl.LOCK_EX)
        #     json.dump(res, output_file)
        #     output_file.write("\n")
        #     fcntl.flock(output_file, fcntl.LOCK_UN)


def run_tasks(arch: str, model_dir: str, model_type: str='fairseq', input_dir: str="data", output_dir: str="data_processed",
              tasks: str=None, train_epochs: int=10, fp16: bool=False, max_sentences: int=1, update_freq: int=16,
              evaluation_only: bool=False, resample: str=None, seed: int=None):
    assert arch in ("roberta_base", "roberta_large", "bert")
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
        runner: TaskRunner = TaskRunner(task, task_id, input_dir, output_dir, model_dir, seed, model_type, arch)
        if not evaluation_only:
            runner.prepare_task(resample)
            runner.train_task(arch, train_epochs, fp16, max_sentences, update_freq)
        score = runner.evaluate_task()
        runner.log_score(task_name, task_id, params, score)


if __name__ == '__main__':
    fire.Fire(run_tasks)
