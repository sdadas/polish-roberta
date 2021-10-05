import logging
import os

import torch
from fairseq import checkpoint_utils

from tasks import BaseTask
from shutil import copy2


class ModelWriter(object):

    def __init__(self, task: BaseTask, model_dir: str):
        self.task = task
        self.model_dir = model_dir
        self.model_name = os.path.basename(model_dir)

    def write_model(self, output_dir: str, minify: bool=True):
        checkpoints_output_dir = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        checkpoint_file = "checkpoint_last.pt" if self.task.spec().no_dev_set else "checkpoint_best.pt"
        checkpoint_path = os.path.join(checkpoints_output_dir, checkpoint_file)
        logging.info("Saving trained model to %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self._copy_model_metafiles(output_dir)
        if minify:
            self._minify_model_checkpoint(checkpoint_path, output_dir)
        else:
            self._copy_model_checkpoint(checkpoint_path, output_dir)

    def _copy_model_metafiles(self, output_dir: str):
        for filename in os.listdir(self.model_dir):
            file_path = os.path.join(self.model_dir, filename)
            output_path = os.path.join(output_dir, filename)
            copy2(file_path, output_path)

    def _copy_model_checkpoint(self, input_path: str, output_dir: str):
        output_path = os.path.join(output_dir, "model.pt")
        copy2(input_path, output_path)

    def _minify_model_checkpoint(self, input_path: str, output_dir: str):
        state = checkpoint_utils.load_checkpoint_to_cpu(input_path)
        del state["last_optimizer_state"]
        output_path = os.path.join(output_dir, "model.pt")
        with open(output_path, "wb") as output_file:
            torch.save(state, output_file)
