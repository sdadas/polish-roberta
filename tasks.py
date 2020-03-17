import logging
import os
from abc import ABC, abstractmethod
from typing import Union, List, Iterable, Callable
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import pearsonr

from utils.normalizer import TextNormalizer


class DataExample(object):

    def __init__(self, inputs: Union[str, List], label: str):
        self.inputs: List[str] = [inputs] if isinstance(inputs, str) else inputs
        self.label: str = label


class TaskSpecification(object):

    def __init__(self, task_dir: str, task_type: str, num_labels: int, num_inputs: int):
        self.task_dir: str = task_dir
        self.output_dir: str = task_dir
        self.task_type: str = task_type
        self.num_labels: int = num_labels
        self.num_inputs: int = num_inputs
        self.evaluation_metric: Callable = self.accuracy if task_type == "classification" else self.pearson
        self.no_dev_set = False

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def pearson(self, y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]

    def f1(self, y_true, y_pred):
        res = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        logging.debug(res)
        return res[2]

    def binary_f1(self, y_true, y_pred):
        y_true = [int(val) for val in y_true]
        y_pred = [int(val) for val in y_pred]
        res = precision_recall_fscore_support(y_true, y_pred, average="binary")
        logging.debug(res)
        return res[2]


class BaseTask(ABC):

    @abstractmethod
    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        raise NotImplementedError

    @abstractmethod
    def spec(self) -> TaskSpecification:
        raise NotImplementedError

    def get_split_path(self, data_path: str, split: str) -> str:
        input_path = os.path.join(data_path, self.spec().task_dir, split + ".txt")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        return input_path

    def read_simple(self, data_path: str, split: str, separator: str=" ", label_first: bool=True, normalize: bool=True):
        label_idx = 0 if label_first else 1
        text_idx = 1 if label_first else 0
        input_path = self.get_split_path(data_path, split)
        normalize_func = lambda val: val
        if normalize:
            normalizer = TextNormalizer()
            normalize_func = lambda val: normalizer.process(val)
        with open(input_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                values = line.split(sep=separator, maxsplit=1)
                label = values[label_idx]
                text = values[text_idx].strip()
                text = normalize_func(text)
                yield DataExample(text, label)

class WCCRSHotelsTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("WCCRS_HOTELS", "classification", 4, 1)

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_simple(data_path, split)

    def spec(self) -> TaskSpecification:
        return self._spec


class WCCRSMedicineTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("WCCRS_MEDICINE", "classification", 4, 1)

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_simple(data_path, split)

    def spec(self) -> TaskSpecification:
        return self._spec


class EightTagsTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("8TAGS", "classification", 8, 1)

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_simple(data_path, split)

    def spec(self) -> TaskSpecification:
        return self._spec


class SICKTask(BaseTask):

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        input_path = self.get_split_path(data_path, split)
        normalizer = TextNormalizer()
        with open(input_path, "r", encoding="utf-8") as input_file:
            for idx, line in enumerate(input_file):
                if idx == 0: continue
                values = line.split("\t")
                input1: str = normalizer.process(values[1].strip())
                input2: str = normalizer.process(values[2].strip())
                relatedness: float = float(values[3].strip())
                entailment: str = values[4].strip()
                yield self.create_example(input1, input2, relatedness, entailment)

    @abstractmethod
    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        raise NotImplementedError


class SICKEntailmentTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("SICK", "classification", 3, 2)
        self._spec.output_dir = "SICK-E"
        self._spec.evaluation_metric = self._spec.accuracy

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        return DataExample([input1, input2], entailment)

    def spec(self) -> TaskSpecification:
        return self._spec


class SICKRelatednessTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("SICK", "regression", 1, 2)
        self._spec.output_dir = "SICK-R"

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        label = "%.5f" % (relatedness / 5.0,)
        return DataExample([input1, input2], label)

    def spec(self) -> TaskSpecification:
        return self._spec


class CDSEntailmentTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("CDS", "classification", 3, 2)
        self._spec.output_dir = "CDS-E"
        self._spec.evaluation_metric = self._spec.accuracy

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        return DataExample([input1, input2], entailment)

    def spec(self) -> TaskSpecification:
        return self._spec


class CDSRelatednessTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("CDS", "regression", 1, 2)
        self._spec.output_dir = "CDS-R"

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        label = "%.5f" % (relatedness / 5.0,)
        return DataExample([input1, input2], label)

    def spec(self) -> TaskSpecification:
        return self._spec


class CBDTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("CBD", "classification", 2, 1)
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        split_name = "training" if split == "train" else split
        file_pattern = "{}_set_clean_only_{}.txt"
        text_path = os.path.join(data_path, self._spec.task_dir, file_pattern.format(split_name, "text"))
        tags_path = os.path.join(data_path, self._spec.task_dir, file_pattern.format(split_name, "tags"))
        normalizer = TextNormalizer(detokenize=False)
        with open(text_path, "r", encoding="utf-8") as text_file, open(tags_path, "r", encoding="utf-8") as tags_file:
            text_lines = text_file.readlines()
            tags_lines = tags_file.readlines()
            assert len(text_lines) == len(tags_lines)
            for idx in range(len(text_lines)):
                text = normalizer.process(text_lines[idx].strip())
                text = text.replace("@anonymized_account", "@ użytkownik")
                label = tags_lines[idx].strip()
                yield DataExample(text, label)

    def spec(self) -> TaskSpecification:
        return self._spec


class PolEmoINTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("POLEMO", "classification", 4, 1)
        self._spec.output_dir = "POLEMO-IN"

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        split = split if split == "train" else f"in-{split}"
        path = self.get_split_path(data_path, split)
        normalizer = TextNormalizer()
        with open(path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                words = line.split()
                label = words[-1]
                text = " ".join(words[0:-1])
                text = text.replace(" em ", "em ").replace(" śmy ", "śmy ").replace(" m ", "m ")
                text = normalizer.process(text)
                yield DataExample(text, label)

    def spec(self) -> TaskSpecification:
        return self._spec


class PolEmoOUTTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("POLEMO", "classification", 4, 1)
        self._spec.output_dir = "POLEMO-OUT"

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        split = split if split == "train" else f"out-{split}"
        path = self.get_split_path(data_path, split)
        normalizer = TextNormalizer()
        with open(path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                words = line.split()
                label = words[-1]
                text = " ".join(words[0:-1])
                text = text.replace(" em ", "em ").replace(" śmy ", "śmy ").replace(" m ", "m ")
                text = normalizer.process(text)
                yield DataExample(text, label)

    def spec(self) -> TaskSpecification:
        return self._spec
