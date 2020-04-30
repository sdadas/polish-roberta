import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Iterable, Callable, Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_absolute_error
from scipy.stats import pearsonr
import csv

from utils.normalizer import TextNormalizer


class DataExample(object):

    def __init__(self, inputs: Union[str, List], label: str):
        self.inputs: List[str] = [inputs] if isinstance(inputs, str) else inputs
        self.label: str = label


class TaskSpecification(object):

    def __init__(self, task_dir: str, task_type: str, num_labels: int, num_inputs: int, group_dir: str=""):
        self.task_dir: str = task_dir
        self.group_dir: str = group_dir
        self.output_dir: str = task_dir
        self.task_type: str = task_type
        self.num_labels: int = num_labels
        self.num_inputs: int = num_inputs
        self.evaluation_metric: Callable = self.accuracy if task_type == "classification" else self.pearson
        self.no_dev_set = False

    def task_path(self) -> str:
        return self.task_dir if not self.group_dir else f"{self.group_dir}/{self.task_dir}"

    def output_path(self) -> str:
        return self.output_dir if not self.group_dir else f"{self.group_dir}/{self.output_dir}"

    def accuracy(self, y_true, y_pred):
        return {"accuracy": accuracy_score(y_true, y_pred)}

    def pearson(self, y_true, y_pred):
        return {"pearson": pearsonr(y_true, y_pred)[0]}

    def f1(self, y_true, y_pred):
        res = precision_recall_fscore_support(y_true, y_pred, average="micro")
        return {"precision": res[0], "recall": res[1], "micro-f1": res[2]}

    def binary_f1(self, y_true, y_pred):
        y_true = [int(val) for val in y_true]
        y_pred = [int(val) for val in y_pred]
        res = precision_recall_fscore_support(y_true, y_pred, average="binary")
        logging.debug(res)
        return {"precision": res[0], "recall": res[1], "binary-f1": res[2]}

    def wmae(self, y_true, y_pred):
        if(isinstance(y_true[0], str)):
            y_true = [float(val) for val in y_true]
            y_pred = [float(val) for val in y_pred]
        y_true_per_class = defaultdict(list)
        y_pred_per_class = defaultdict(list)
        for yt, yp in zip(y_true, y_pred):
            y_true_per_class[yt].append(yt)
            y_pred_per_class[yt].append(yp)
        mae = []
        for clazz in y_true_per_class.keys():
            yt = y_true_per_class[clazz]
            yp = y_pred_per_class[clazz]
            mae.append(mean_absolute_error(yt, yp))
        mae_avg = sum(mae) / len(mae)
        return {"wmae": mae_avg, "1-wmae": 1 - mae_avg}


class BaseTask(ABC):

    @abstractmethod
    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        raise NotImplementedError

    def spec(self) -> TaskSpecification:
        return self.__getattribute__("_spec")

    def get_split_path(self, data_path: str, split: str) -> str:
        input_path = os.path.join(data_path, self.spec().task_path(), split + ".txt")
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


class WCCRSMedicineTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("WCCRS_MEDICINE", "classification", 4, 1)

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_simple(data_path, split)


class EightTagsTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("8TAGS", "classification", 8, 1)

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_simple(data_path, split)


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

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        return DataExample([input1, input2], entailment)


class SICKRelatednessTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("SICK", "regression", 1, 2)
        self._spec.output_dir = "SICK-R"

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        label = "%.5f" % (relatedness / 5.0,)
        return DataExample([input1, input2], label)

    def format_output(self, value: float):
        return "%.2f" % (value * 5,)

    def postprocess_prediction(self, value: float):
        return min((max(value, 0.0)), 1.0)


class CDSEntailmentTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("CDS", "classification", 3, 2)
        self._spec.output_dir = "CDS-E"

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        return DataExample([input1, input2], entailment)


class CDSRelatednessTask(SICKTask):

    def __init__(self):
        self._spec = TaskSpecification("CDS", "regression", 1, 2)
        self._spec.output_dir = "CDS-R"

    def create_example(self, input1: str, input2: str, relatedness: float, entailment: str):
        label = "%.5f" % (relatedness / 5.0,)
        return DataExample([input1, input2], label)

    def format_output(self, value: float):
        return "%.2f" % (value * 5,)

    def postprocess_prediction(self, value: float):
        return min((max(value, 0.0)), 1.0)


class CBDTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("CBD", "classification", 2, 1)
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        split_name = "training" if split == "train" else split
        file_pattern = "{}_set_clean_only_{}.txt"
        text_path = os.path.join(data_path, self._spec.task_path(), file_pattern.format(split_name, "text"))
        tags_path = os.path.join(data_path, self._spec.task_path(), file_pattern.format(split_name, "tags"))
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


class KLEJTask(BaseTask):

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        input_path = os.path.join(data_path, self.spec().task_path(), split + ".tsv")
        normalizer = self.normalizer()
        with open(input_path, "r", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file, dialect="excel-tab")
            for row in reader:
                yield self.create_example(row, normalizer)

    def normalizer(self):
        return TextNormalizer()

    @abstractmethod
    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        raise NotImplementedError


class KLEJCBDTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("CBD", "classification", 2, 1, "KLEJ")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def normalizer(self) -> TextNormalizer:
        return TextNormalizer(detokenize=False)

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text = normalizer.process(row["sentence"].strip())
        text = text.replace("@anonymized_account", "@ użytkownik")
        return DataExample(text, row["target"].strip())


class KLEJDYKTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("DYK", "classification", 2, 2, "KLEJ")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text1 = row["question"].strip()
        text2 = row["answer"].strip()
        return DataExample([text1, text2], row["target"].strip())


class KLEJPSCTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("PSC", "classification", 2, 2, "KLEJ")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text1 = row["extract_text"].strip()
        text2 = row["summary_text"].strip()
        return DataExample([text1, text2], row["label"].strip())


class KLEJPolEmoINTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("POLEMO2.0-IN", "classification", 4, 1, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text= normalizer.process(row["sentence"].strip())
        return DataExample(text, row["target"].strip())


class KLEJPolEmoOUTTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("POLEMO2.0-OUT", "classification", 4, 1, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text= normalizer.process(row["sentence"].strip())
        return DataExample(text, row["target"].strip())


class KLEJCDSEntailmentTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("CDSC-E", "classification", 3, 2, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text1 = normalizer.process(row["sentence_A"].strip())
        text2 = normalizer.process(row["sentence_B"].strip())
        return DataExample([text1, text2], row["entailment_judgment"].strip())


class KLEJCDSRelatednessTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("CDSC-R", "regression", 1, 2, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text1 = normalizer.process(row["sentence_A"].strip())
        text2 = normalizer.process(row["sentence_B"].strip())
        score = float(row["relatedness_score"])
        return DataExample([text1, text2], "%.5f" % (score / 5.0,))

    def format_output(self, value: float):
        return "%.2f" % (value * 5,)

    def postprocess_prediction(self, value: float):
        return min((max(value, 0)), 1)


class KLEJNKJPTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("NKJP-NER", "classification", 6, 1, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text = normalizer.process(row["sentence"].strip())
        return DataExample(text, row["target"].strip())


class KLEJECRRegressionTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("ECR", "regression", 1, 1, "KLEJ")
        self._spec.evaluation_metric = self._spec.wmae

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text = row["text"].strip()
        score = float(row["rating"]) - 1.0
        return DataExample(text, "%.5f" % (score / 4.0,))

    def format_output(self, value: float):
        return "%.2f" % (1 + value * 4,)

    def postprocess_prediction(self, value: float):
        score = min(max(1 + value * 4, 0), 5)
        score = round(score)
        return (score - 1.0) / 4.0


class KLEJECRClassificationTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("ECR", "classification", 5, 1, "KLEJ")
        self._spec.evaluation_metric = self._spec.wmae

    def create_example(self, row: Dict, normalizer: TextNormalizer) -> DataExample:
        text = row["text"].strip()
        return DataExample(text, row["rating"].strip())