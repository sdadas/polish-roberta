import os
import re

import torch
from fairseq import checkpoint_utils
from typing import Dict

from tasks import BaseTask


class BlockReinitializer(object):

    def __init__(self, task: BaseTask, model_dir: str, reinit_layers: int=5):
        self.task = task
        self.model_dir = model_dir
        self.reinit_layers = reinit_layers
        self.model_name = os.path.basename(model_dir)

    def reinit(self):
        input_path = os.path.join(self.model_dir, "model.pt")
        state = checkpoint_utils.load_checkpoint_to_cpu(input_path)
        args = state.get("args")
        model: Dict = state.get("model")
        pre_trained_layers = args.encoder_layers - self.reinit_layers
        for layer_name, layer_weight in model.items():
            match = re.search(r"layers.(\d+).", layer_name)
            if match is not None:
                layer_num = int(match.group(1))
                if layer_num >= pre_trained_layers:
                    self._reinit_weight(layer_name, layer_weight)
        self._save_model(state)

    def _reinit_weight(self, layer_name: str, layer_weight):
        if layer_name.endswith("bias"):
            layer_weight.zero_()
        elif layer_name.endswith("weight"):
            layer_weight.normal_(mean=0.0, std=0.02)
        else: raise Exception(f"Unknown layer passed to reinit '{layer_name}'")

    def _save_model(self, state):
        checkpoints_output_dir = os.path.join("checkpoints", self.model_name, self.task.spec().output_path())
        os.makedirs(checkpoints_output_dir, exist_ok=True)
        output_path = os.path.join(checkpoints_output_dir, "checkpoint_last.pt")
        with open(output_path, "wb") as output_file:
            torch.save(state, output_file)
