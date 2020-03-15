import logging
import os
import subprocess

from tasks import BaseTask


class TaskTrainer(object):

    def __init__(self, task: BaseTask, data_path: str, model_path: str, train_size: int,
                 checkpoint: str="checkpoint_best.pt", arch: str="roberta_large", fp16: bool=False):
        self.task: BaseTask = task
        self.train_size: int = train_size
        self.data_path: str = data_path
        self.task_data_path: str = os.path.join(self.data_path, task.spec().output_dir + "-bin")
        self.model_path: str = model_path
        self.checkpoint: str = checkpoint
        self.arch: str = arch
        self.learning_rate = "1e-5"
        self.fp16 = fp16

    def train(self, max_sentences: int=1, update_freq: int=16, train_epochs: int=10):
        self._run_fairseq_train(1, 16, max_epoch=train_epochs)

    def _run_fairseq_train(self, max_sentences: int=16, update_freq: int=1, max_epoch: int=10):
        batch_size: int = max_sentences * update_freq
        total_updates: int = int((self.train_size * max_epoch) / batch_size)
        warmup_updates: int = int(total_updates / 16.67)
        cmd = [
            "fairseq-train",
            self.task_data_path,
            "--restore-file", os.path.join(self.model_path, self.checkpoint),
            "--max-positions", "512",
            "--max-sentences", str(max_sentences),
            "--update-freq", str(update_freq),
            "--max-tokens", "4400",
            "--task", "sentence_prediction",
            "--reset-optimizer",
            "--reset-dataloader",
            "--reset-meters",
            "--required-batch-size-multiple", "1",
            "--init-token", "0",
            "--separator-token", "2",
            "--arch", self.arch,
            "--criterion", "sentence_prediction",
            "--num-classes", str(self.task.spec().num_labels),
            "--dropout", "0.1",
            "--attention-dropout", "0.1",
            "--weight-decay", "0.1",
            "--optimizer", "adam",
            "--adam-betas", "(0.9, 0.98)",
            "--adam-eps", "1e-06",
            "--clip-norm", "0.0",
            "--lr-scheduler", "polynomial_decay",
            "--lr", self.learning_rate,
            "--total-num-update", str(total_updates),
            "--warmup-updates", str(warmup_updates),
            "--max-epoch", str(max_epoch),
            "--find-unused-parameters",
            "--log-format", "simple",
            "--log-interval", "5",
            "--save-dir", os.path.join("checkpoints", self.task.spec().output_dir)
        ]
        task_type = self.task.spec().task_type
        if task_type == "classification":
            cmd.extend([
                "--best-checkpoint-metric", "accuracy",
                "--maximize-best-checkpoint-metric"
            ])
        elif task_type == "regression":
            cmd.extend([
                "--regression-target",
                "--best-checkpoint-metric", "loss"
            ])
        if self.fp16:
            cmd.extend([
                "--fp16",
                "--fp16-init-scale", "4",
                "--threshold-loss-scale", "1",
                "--fp16-scale-window", "128"
            ])
        logging.info("running %s", cmd.__repr__())
        subprocess.run(cmd)