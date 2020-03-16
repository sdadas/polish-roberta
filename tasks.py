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
        self.evaluation_metric: Callable = self.f1 if task_type == "classification" else self.pearson

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def pearson(self, y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]

    def f1(self, y_true, y_pred):
        average = "binary" if self.num_labels <= 2 else "weighted"
        return precision_recall_fscore_support(y_true, y_pred, average=average)[2]


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

        with open(input_path, "r", encoding="utf-8") as input_file:
            for idx, line in enumerate(input_file):
                if idx == 0: continue
                values = line.split("\t")
                input1: str = values[1].strip()
                input2: str = values[2].strip()
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