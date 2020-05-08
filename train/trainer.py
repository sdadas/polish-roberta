import logging
import os
import subprocess
import random
import shutil
import torch

from simpletransformers.classification import ClassificationModel

from tasks import BaseTask


class TaskTrainer(object):

    def __init__(self, task: BaseTask, data_path: str, model_path: str, train_size: int, checkpoint: str="model.pt",
                 arch: str="roberta_large", fp16: bool=False, model_type: str='fairseq'):
        self.task: BaseTask = task
        self.train_size: int = train_size
        self.data_path: str = data_path
        self.task_data_path: str = os.path.join(self.data_path, task.spec().output_path() + "-bin" if model_type == 'fairseq' else '')
        self.model_path: str = model_path
        self.model_name: str = os.path.basename(model_path)
        self.model_type: str = model_type
        self.checkpoint: str = checkpoint
        self.arch: str = arch
        self.learning_rate = "1e-5"
        self.fp16 = fp16

    def train(self, max_sentences: int=1, update_freq: int=16, train_epochs: int=10, seed: int=None):
        if self.model_type == 'fairseq':
            self._run_fairseq_train(seed, max_sentences=max_sentences, update_freq=update_freq, max_epoch=train_epochs)
        elif self.model_type == 'transformers':
            self._run_transformers_train(seed, max_sentences=max_sentences, update_freq=update_freq, max_epoch=train_epochs)

    def _remove_previous_checkpoints(self, checkpoint_path: str):
        checkpoint_last = os.path.join(checkpoint_path, "checkpoint_last.pt")
        if os.path.exists(checkpoint_last): os.remove(checkpoint_last)
        checkpoint_best = os.path.join(checkpoint_path, "checkpoint_best.pt")
        if os.path.exists(checkpoint_best): os.remove(checkpoint_best)

    def _run_fairseq_train(self, seed: int, max_sentences: int=16, update_freq: int=1, max_epoch: int=10):
        if seed is None: seed = random.randint(0, 1_000_000)
        batch_size: int = max_sentences * update_freq
        total_updates: int = int((self.train_size * max_epoch) / batch_size)
        warmup_updates: int = int(total_updates / 16.67)
        restore_file = os.path.join(self.model_path, self.checkpoint)
        assert os.path.exists(restore_file)
        checkpoint_path = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        self._remove_previous_checkpoints(checkpoint_path)
        cmd = [
            "fairseq-train",
            self.task_data_path,
            "--restore-file", restore_file,
            "--seed", str(seed),
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
            "--save-dir", checkpoint_path,
            "--no-epoch-checkpoints"
        ]
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
        logging.info("running %s", cmd.__repr__())
        subprocess.run(cmd)

    def _run_transformers_train(self, seed: int, max_sentences: int=16, update_freq: int=1, max_epoch: int=10):
        if seed is None: seed = random.randint(0, 1_000_000)
        batch_size: int = max_sentences * update_freq
        total_updates: int = int((self.train_size * max_epoch) / batch_size)
        warmup_updates: int = int(total_updates / 16.67)
        checkpoint_path = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        if os.path.exists(checkpoint_path): shutil.rmtree(checkpoint_path)

        args = {
            "output_dir": checkpoint_path,
            "cache_dir": 'cache',
            "best_model_dir": os.path.join(checkpoint_path, 'best_checkpoint'),
            'learning_rate': float(self.learning_rate),
            'weight_decay': 0.1,
            'adam_epsilon': 1e-06,
            'save_eval_checkpoints': not self.task.spec().no_dev_set,
            'evaluate_during_training': not self.task.spec().no_dev_set,
            'train_batch_size': batch_size,
            'eval_batch_size': batch_size,
            'num_train_epochs': max_epoch,
            'max_seq_length': 100,
            'do_lower_case': False,
            'no_cache': True,
            'save_model_every_epoch': True,
            'tensorboard_dir': None,
            'overwrite_output_dir': True,
            'reprocess_input_data': False,
            'process_count': 1,
            'n_gpu': 4,
            'silent': False,
            'use_multiprocessing': True,
            "warmup_steps": warmup_updates,
            'manual_seed': seed,
            'fp16': False,
            'max_steps': total_updates,
            'gradient_accumulation_steps': update_freq
        }

        if self.task.spec().task_type == "classification":
            model = ClassificationModel(self.arch, self.model_path, num_labels=self.task.spec().num_labels,
                                        use_cuda=torch.cuda.is_available(), args=args)
        else:
            raise Exception(f'Unhandled task type exception: {self.task.spec().task_type}')

        train_df = self.task.read_csv(self.data_path, 'train', label_first=False, normalize=False)
        eval_df = None
        if not self.task.spec().no_dev_set:
            eval_df = self.task.read_csv(self.data_path, 'dev', label_first=False, normalize=False)

        model.train_model(train_df, eval_df=eval_df, multi_label=False, show_running_loss=True)
