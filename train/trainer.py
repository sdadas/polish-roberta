import logging
import os
import random
import subprocess
import glob
from typing import List

from fairseq import options, __version__ as fairseq_verison

from tasks import BaseTask


class TaskTrainer(object):

    def __init__(self, task: BaseTask, data_path: str, model_path: str, train_size: int, checkpoint: str="model.pt",
                 arch: str="roberta_large", fp16: bool=False, ddp_backend: str="pytorch_ddp", cpu_offload: bool=False):
        self.task: BaseTask = task
        self.train_size: int = train_size
        self.data_path: str = data_path
        self.task_data_path: str = os.path.join(self.data_path, task.spec().output_path() + "-bin")
        self.model_path: str = model_path
        self.model_name: str = os.path.basename(model_path)
        self.checkpoint: str = checkpoint
        self.arch: str = arch
        self.learning_rate = "1e-5"
        self.fp16 = fp16
        self.ddp_backend = ddp_backend
        self.cpu_offload = cpu_offload if ddp_backend == "fully_sharded" else False
        self.cli_path = os.environ.get("CLI_PATH", "")

    def train(self, max_sentences: int=1, update_freq: int=16, train_epochs: int=10, seed: int=None):
        self._run_fairseq_train(seed, max_sentences=max_sentences, update_freq=update_freq, max_epoch=train_epochs)

    def _remove_previous_checkpoints(self, checkpoint_path: str):
        pattern = os.path.join(checkpoint_path, "checkpoint*")
        found_checkpoint_files = glob.glob(pattern)
        for checkpoint in found_checkpoint_files:
            logging.info("Removing old checkpoint file %s", checkpoint)
            os.remove(checkpoint)

    def _run_fairseq_train(self, seed: int, max_sentences: int=16, update_freq: int=1, max_epoch: int=10):
        if seed is None: seed = random.randint(0, 1_000_000)
        batch_size: int = max_sentences * update_freq
        total_updates: int = int((self.train_size * max_epoch) / batch_size)
        warmup_updates: int = int(total_updates / 16.67)
        restore_file = os.path.join(self.model_path, self.checkpoint)
        assert os.path.exists(restore_file)
        checkpoint_path = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        self._remove_previous_checkpoints(checkpoint_path)
        arch = "roberta_prenorm" if self.arch.startswith("xlmr") else self.arch
        cmd = [
            self.task_data_path,
            "--restore-file", restore_file,
            "--seed", str(seed),
            "--max-tokens", "4400",
            "--task", "sentence_prediction",
            "--reset-optimizer",
            "--reset-dataloader",
            "--reset-meters",
            "--required-batch-size-multiple", "1",
            "--init-token", "0",
            "--separator-token", "2",
            "--arch", arch,
            "--criterion", "sentence_prediction",
            "--num-classes", str(self.task.spec().num_labels),
            "--dropout", "0.1",
            "--attention-dropout", "0.1",
            "--weight-decay", "0.1",
            "--optimizer", "adam" if not self.cpu_offload else "cpu_adam",
            "--adam-betas", "(0.9,0.98)",
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
            "--save-dir", checkpoint_path,
            "--no-epoch-checkpoints",
            "--ddp-backend", self.ddp_backend
        ]
        if fairseq_verison == "0.9.0":
            cmd.extend([
                "--max-sentences", str(max_sentences),
                "--update-freq", str(update_freq)
            ])
        else:
            cmd.extend([
                "--batch-size", str(max_sentences),
                "--update-freq", str(update_freq)
            ])

        self._arch_specific_options(cmd)
        if self.task.spec().no_dev_set:
            cmd.extend([
                "--disable-validation",
                "--valid-subset", "train"
            ])
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
        if self.cpu_offload:
            cmd.extend([
                "--cpu-offload",
                "--no-reshard-after-forward",
                "--no-save-optimizer-state",
                "--use-sharded-state"
            ])
        logging.info("running %s", cmd.__repr__())
        self._run_training(cmd)

    def _run_training(self, cmd: List[str]):
        if self.ddp_backend == "fully_sharded":
            self._run_training_legacy(cmd)
        else:
            try:
                from fairseq_cli.train import main
                parser = options.get_training_parser()
                if self.arch.startswith("bart"):
                    parser.add_argument("--max-positions", type=int)
                args = options.parse_args_and_arch(parser, input_args=cmd)
                if self.arch.startswith("bart"):
                    args.spectral_norm_classification_head = False
                main(args)
            except ImportError:
                self._run_training_legacy(cmd)


    def _run_training_legacy(self, cmd: List[str]):
        cmd.insert(0, f"{self.cli_path}fairseq-train")
        subprocess.run(cmd)

    def _arch_specific_options(self, cmd: List[str]):
        if self.arch.startswith("bart"):
            cmd.extend([
                "--add-prev-output-tokens",
                "--layernorm-embedding",
                "--share-all-embeddings",
                "--share-decoder-input-output-embed",
                "--init-token", "0",
                "--max-positions", "1024"
            ])
        elif self.arch == "xlmr.xl":
            cmd.extend([
                "--encoder-attention-heads", "32",
                "--encoder-embed-dim", "2560",
                "--encoder-ffn-embed-dim", "10240",
                "--encoder-layers", "36",
                "--max-positions", "512"
            ])
        else:
            cmd.extend([
                "--max-positions", "512",
            ])