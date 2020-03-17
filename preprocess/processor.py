import logging
import multiprocessing
import os
import shutil
import subprocess
from typing import TextIO, List

from preprocess.spm_encode import spm_encode
from tasks import BaseTask


class TaskProcessor(object):

    def __init__(self, task: BaseTask, data_path: str, output_path: str, model_path: str):
        self.task: BaseTask = task
        self.data_path: str = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.task_output_path = os.path.join(self.output_path, task.spec().output_dir)
        if not os.path.exists(self.task_output_path):
            os.makedirs(self.task_output_path, exist_ok=True)

    def prepare(self):
        self._prepare_split("train")
        self._prepare_split("dev")
        self._prepare_split("test")
        self._spm_encode()
        self._fairseq_preprocess()

    def _prepare_split(self, split: str):
        if split == "dev" and self.task.spec().no_dev_set: return
        num_inputs: int = self.task.spec().num_inputs
        num_outputs: int = num_inputs + 1
        outputs: List[TextIO] = self._open_outputs(split, num_inputs)
        for record in self.task.read(self.data_path, split):
            rec_out = [record.label]
            rec_out.extend(record.inputs)
            assert len(rec_out) == num_outputs
            for idx in range(num_outputs):
                outputs[idx].write(rec_out[idx])
                outputs[idx].write("\n")
        for output in outputs:
            output.close()

    def _open_outputs(self, split: str, num_inputs: int) -> List[TextIO]:
        res = []
        res.append(open(os.path.join(self.task_output_path, split + ".label"), "w", encoding="utf-8"))
        for idx in range(num_inputs):
            file_name = split + ".raw.input" + str(idx)
            res.append(open(os.path.join(self.task_output_path, file_name), "w", encoding="utf-8"))
        return res

    def _spm_encode(self):
        for file in os.listdir(self.task_output_path):
            if ".raw.input" in file:
                self._spm_encode_file(os.path.join(self.task_output_path, file))

    def _spm_encode_file(self, file_path: str):
        output_path = file_path.replace(".raw.input", ".input")
        spm_model = os.path.join(self.model_path, "sentencepiece.model")
        spm_encode(file_path, output_path, spm_model, threads=1, encode_ids=False, max_length=510)

    def _fairseq_preprocess(self):
        num_inputs: int = self.task.spec().num_inputs
        output_path: str = self.task_output_path + "-bin"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        for input_idx in range(num_inputs):
            self._fairseq_preprocess_input(input_idx, output_path)
        if self.task.spec().task_type == "regression": self._copy_labels(output_path)
        else: self._fairseq_preprocess_labels(output_path)

    def _copy_labels(self, output_path: str):
        destdir = os.path.join(output_path, "label")
        os.mkdir(destdir)
        shutil.copy(os.path.join(self.task_output_path, "train.label"), os.path.join(destdir, "train.label"))
        shutil.copy(os.path.join(self.task_output_path, "dev.label"), os.path.join(destdir, "valid.label"))

    def _fairseq_preprocess_input(self, input_idx: int, output_path: str):
        input_name = f"input{input_idx}"
        destdir = os.path.join(output_path, input_name)
        self._run_fairseq_preprocess(input_name, destdir)

    def _fairseq_preprocess_labels(self, output_path: str):
        input_name = "label"
        destdir = os.path.join(output_path, input_name)
        self._run_fairseq_preprocess(input_name, destdir)

    def _run_fairseq_preprocess(self, input_name: str, destdir: str):
        cpus = multiprocessing.cpu_count()
        cmd = ["fairseq-preprocess", "--only-source", "--workers", str(cpus), "--destdir", destdir]
        for split in ("train", "dev", "test"):
            if input_name == "label" and split == "test": continue
            if split == "dev" and self.task.spec().no_dev_set: continue
            file_name = split + "." + input_name
            pref = os.path.join(self.task_output_path, file_name)
            option = "--validpref" if split == "dev" else f"--{split}pref"
            cmd.append(option)
            cmd.append(pref)
        if not input_name == "label":
            dict_path: str = os.path.join(self.model_path, "dict.txt")
            cmd.append("--srcdict")
            cmd.append(dict_path)
        logging.info("running %s", cmd.__repr__())
        subprocess.run(cmd)