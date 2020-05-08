import logging
import multiprocessing
import os
import shutil
import subprocess
from typing import TextIO, List, Optional, Dict
import random
import string
import pandas as pd

from preprocess.spm_encode import spm_encode
from tasks import BaseTask


class TaskProcessor(object):

    def __init__(self, task: BaseTask, data_path: str, output_path: str, model_path: str, resample: str, model_format: str = 'fairseq'):
        self.task: BaseTask = task
        self.data_path: str = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.task_output_path = os.path.join(self.output_path, task.spec().output_path())
        self.resample = self._parse_resample_string(resample)
        self.model_format = model_format
        if not os.path.exists(self.task_output_path):
            os.makedirs(self.task_output_path, exist_ok=True)

    def _parse_resample_string(self, resample) -> Optional[Dict]:
        if not resample: return None
        values = [val.strip() for val in resample.split(",")]
        res = {}
        for value in values:
            if not ":" in value: continue
            keyval = value.split(":", maxsplit=2)
            key = keyval[0].strip()
            val = float(keyval[1].strip())
            res[key] = val
        return res

    def prepare(self):
        self._prepare_split("train")
        self._prepare_split("dev")
        self._prepare_split("test")
        if self.model_format == 'fairseq':
            self._spm_encode()
            self._fairseq_preprocess()
        elif self.model_format == 'transformers':
            self._transformers_preprocess()

    def _resampling_wrapper(self, data_path, split):
        resampled = []
        for record in self.task.read(data_path, split):
            label = record.label
            resampling_value = self.resample.get(label)
            if resampling_value is None:
                yield record
            elif resampling_value < 1:
                if random.random() <= resampling_value:
                    yield record
            elif resampling_value > 1:
                yield record
                decimal_part = resampling_value % 1
                whole_part = int(resampling_value)
                for idx in range(whole_part):
                    if idx == 0: yield record
                    else: resampled.append(record)
                if random.random() <= decimal_part:
                    resampled.append(record)
            else:
                yield record
        for record in resampled:
            yield record

    def _prepare_split(self, split: str):
        if split == "dev" and self.task.spec().no_dev_set: return
        num_inputs: int = self.task.spec().num_inputs
        num_outputs: int = num_inputs + 1
        outputs: List[TextIO] = self._open_outputs(split, num_inputs)
        reader = self._resampling_wrapper if split == "train" and self.resample is not None else self.task.read
        for record in reader(self.data_path, split):
            rec_out = [record.label]
            rec_out.extend(record.inputs)
            assert len(rec_out) == num_outputs, rec_out
            for idx in range(num_outputs):
                if rec_out[idx] is None and idx == 0 and split == "test":
                    continue # skip missing test labels
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
        spm_model = os.path.join(self.model_path, "sentencepiece.bpe.model")
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

    def _transformers_preprocess(self):
        alphabet = list(string.ascii_lowercase)
        num_inputs: int = self.task.spec().num_inputs
        for split in ("train", "dev", "test"):
            processed = {}
            if split == 'dev' and self.task.spec().no_dev_set: continue
            for input_idx in range(num_inputs):
                input_raw_path = os.path.join(self.task_output_path, f'{split}.raw.input{input_idx}')
                processed[f'text_{alphabet[input_idx]}'] = self._transformers_preprocess_raw(input_raw_path)
            labels_raw_path = os.path.join(self.task_output_path, f'{split}.label')
            processed['labels'] = self._transformers_preprocess_raw(labels_raw_path, as_int=True)
            pd.DataFrame(processed).to_csv(os.path.join(self.task_output_path, f'{split}.tsv'), sep='\t', index=False)

    def _transformers_preprocess_raw(self, raw_file_path: str, as_int: bool = False) -> List:
        input_io = open(raw_file_path, 'r', newline=None, encoding='utf-8')
        processed = list(map(str.strip, input_io.readlines()))
        if as_int:
            labels = set()
            [labels.add(item) for item in processed]
            labels = list(labels)
            labels.sort()
            processed = list(map(labels.index, processed))
        return processed

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