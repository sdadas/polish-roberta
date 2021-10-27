import json
import random
import os
from itertools import chain
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Iterable, Callable, Dict, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_absolute_error, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from utils.normalizer import TextNormalizer


class DataExample(object):

    def __init__(self, inputs: Union[str, List], label: Optional[str]):
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
        self.evaluation_metric: Callable = self.accuracy if task_type == "classification" else self.corr
        self.no_dev_set = False

    def to_json(self):
        return {
            "dir": self.task_dir,
            "group": self.group_dir,
            "type": self.task_type,
            "num_labels": self.num_labels,
            "num_inputs": self.num_inputs,
            "metric": self.evaluation_metric.__name__
        }

    def task_path(self) -> str:
        return self.task_dir if not self.group_dir else f"{self.group_dir}/{self.task_dir}"

    def output_path(self) -> str:
        return self.output_dir if not self.group_dir else f"{self.group_dir}/{self.output_dir}"

    def accuracy(self, y_true, y_pred):
        return {"accuracy": accuracy_score(y_true, y_pred)}

    def corr(self, y_true, y_pred):
        return {"pearson": pearsonr(y_true, y_pred)[0], "spearman": spearmanr(y_true, y_pred)[0]}

    def mcc(self, y_true, y_pred):
        return {"mcc": matthews_corrcoef(y_true, y_pred)}

    def f1(self, y_true, y_pred):
        res = precision_recall_fscore_support(y_true, y_pred, average="micro")
        acc = accuracy_score(y_true, y_pred)
        return {"precision": res[0], "recall": res[1], "micro-f1": res[2], "accuracy": acc}

    def binary_f1(self, y_true, y_pred):
        y_true = [int(val) for val in y_true]
        y_pred = [int(val) for val in y_pred]
        res = precision_recall_fscore_support(y_true, y_pred, average="binary")
        acc = accuracy_score(y_true, y_pred)
        return {"precision": res[0], "recall": res[1], "binary-f1": res[2], "accuracy": acc}

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

    def get_split_path(self, data_path: str, split: str, extension: str="txt") -> str:
        input_path = os.path.join(data_path, self.spec().task_path(), split + "." + extension)
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


class CrossValidatedTask(BaseTask):

    def __init__(self, wrapped_task: BaseTask, num_folds: int=4, seed: int=None):
        self.wrapped_task: BaseTask = wrapped_task
        self.num_folds = num_folds
        self.folds = None
        self._spec = wrapped_task.spec()
        self.set_fold(0)
        self.seed = seed

    def set_fold(self, fold: int):
        self.fold = fold
        self._spec.output_dir = f"{self._spec.task_dir}-fold{self.fold}"

    def _read_folds(self, data_path: str):
        if self.seed is not None: random.seed(self.seed)
        data: List[DataExample] = []
        for record in self.wrapped_task.read(data_path, "train"):
            data.append(record)
        random.shuffle(data)
        folds = [[] for _ in range(self.num_folds)]
        for idx, record in enumerate(data):
            fold_idx = idx % self.num_folds
            folds[fold_idx].append(record)
        return folds

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        if self.folds is None:
            self.folds = self._read_folds(data_path)
        if split == "dev":
            return self.wrapped_task.read(data_path, split)
        elif split == "train":
            folds = [self.folds[idx] for idx in range(self.num_folds) if idx != self.fold]
            return chain(*folds)
        elif split == "test":
            return [rec for rec in self.folds[self.fold]]

    @staticmethod
    def cv_folds(wrapped_task: BaseTask, num_folds: int=4, seed: int=None) -> Iterable[BaseTask]:
        task = CrossValidatedTask(wrapped_task, num_folds, seed)
        for fold in range(num_folds):
            task.set_fold(fold)
            yield task


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
        has_target = True
        if split == "test" and not os.path.exists(input_path):
            input_path = os.path.join(data_path, self.spec().task_path(), split + "_features.tsv")
            has_target = False
        normalizer = self.normalizer()
        with open(input_path, "r", encoding="utf-8") as input_file:
            header = input_file.readline().strip().split("\t")
            for line in input_file:
                values = [val.strip() for val in line.split("\t")]
                assert len(header) == len(values), values
                row = {key: val for key, val in zip(header, values)}
                yield self.create_example(row, normalizer, has_target)

    def normalizer(self):
        return TextNormalizer()

    @abstractmethod
    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        raise NotImplementedError


class KLEJCBDTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("CBD", "classification", 2, 1, "KLEJ")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def normalizer(self) -> TextNormalizer:
        return TextNormalizer(detokenize=False)

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text = normalizer.process(row["sentence"].strip())
        text = text.replace("@anonymized_account", "@ użytkownik")
        return DataExample(text, row["target"].strip() if has_target else None)


class KLEJDYKTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("DYK", "classification", 2, 2, "KLEJ")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def normalizer(self) -> TextNormalizer:
        return TextNormalizer(detokenize=False)

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text1 = row["question"].strip()
        text2 = row["answer"].strip()
        return DataExample([text1, text2], row["target"].strip() if has_target else None)


class KLEJPSCTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("PSC", "classification", 2, 2, "KLEJ")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def normalizer(self) -> TextNormalizer:
        return TextNormalizer(detokenize=False)

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text1 = row["extract_text"].strip()
        text2 = row["summary_text"].strip()
        return DataExample([text1, text2], row["label"].strip() if has_target else None)


class KLEJPolEmoINTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("POLEMO2.0-IN", "classification", 4, 1, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text= normalizer.process(row["sentence"].strip())
        return DataExample(text, row["target"].strip() if has_target else None)


class KLEJPolEmoOUTTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("POLEMO2.0-OUT", "classification", 4, 1, "KLEJ")
        self.labels = ("__label__meta_minus_m", "__label__meta_plus_m", "__label__meta_amb", "__label__meta_zero")

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text= normalizer.process(row["sentence"].strip())
        for label in self.labels:
            if label in text:
                print(text)
        return DataExample(text, row["target"].strip() if has_target else None)


class KLEJCDSEntailmentTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("CDSC-E", "classification", 3, 2, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text1 = normalizer.process(row["sentence_A"].strip())
        text2 = normalizer.process(row["sentence_B"].strip())
        return DataExample([text1, text2], row["entailment_judgment"].strip() if has_target else None)


class KLEJCDSRelatednessTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("CDSC-R", "regression", 1, 2, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text1 = normalizer.process(row["sentence_A"].strip())
        text2 = normalizer.process(row["sentence_B"].strip())
        if has_target:
            score = float(row["relatedness_score"])
            score = "%.5f" % (score / 5.0,)
        else: score = None
        return DataExample([text1, text2], score)

    def format_output(self, value: float):
        return "%.2f" % (value * 5,)

    def postprocess_prediction(self, value: float):
        return min((max(value, 0)), 1)


class KLEJNKJPTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("NKJP-NER", "classification", 6, 1, "KLEJ")

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text = normalizer.process(row["sentence"].strip())
        return DataExample(text, row["target"].strip() if has_target else None)


class KLEJECRRegressionTask(KLEJTask):

    def __init__(self):
        self._spec = TaskSpecification("ECR", "regression", 1, 1, "KLEJ")
        self._spec.evaluation_metric = self._spec.wmae

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text = row["text"].strip()
        if has_target:
            score = float(row["rating"]) - 1.0
            score = "%.5f" % (score / 4.0,)
        else: score = None
        return DataExample(text, score)

    def normalizer(self) -> TextNormalizer:
        return TextNormalizer(detokenize=False)

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

    def normalizer(self) -> TextNormalizer:
        return TextNormalizer(detokenize=False)

    def create_example(self, row: Dict, normalizer: TextNormalizer, has_target: bool) -> DataExample:
        text = row["text"].strip()
        return DataExample(text, row["rating"].strip() if has_target else None)


class GLUETask(BaseTask):

    def read_data_file(self, data_path: str, split: str, file_name: str, has_header: bool):
        input_path = os.path.join(data_path, self.spec().task_path(), file_name)
        normalizer = self.normalizer()
        with open(input_path, "r", encoding="utf-8") as input_file:
            if has_header:
                _ = input_file.readline().strip().split("\t")
            for line in input_file:
                row = [val.strip() for val in line.split("\t")]
                yield self.create_example(row, normalizer, split)

    def normalizer(self):
        return TextNormalizer(detokenize=False, lang="en")

    @abstractmethod
    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        raise NotImplementedError


class GLUECoLATask(GLUETask):

    def __init__(self) -> None:
        self._spec = TaskSpecification("CoLA", "classification", 2, 1, "GLUE")
        self._spec.evaluation_metric = self._spec.mcc

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, split + ".tsv", split != "test")

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text = row[3 if split != "test" else 1].strip()
        label = row[1] if split != "test" else None
        return DataExample(text, label)


class GLUEQQPTask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("QQP", "classification", 2, 2, "GLUE")
        self._spec.evaluation_metric = self._spec.binary_f1

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, split + ".tsv", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[3 if split != "test" else 1].strip()
        text2 = row[4 if split != "test" else 2].strip()
        label = row[5] if split != "test" else None
        return DataExample([text1, text2], label)


class GLUEMNLIMatchedTask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("MNLI", "classification", 3, 2, "GLUE")
        self._spec.output_dir = "MNLI-Matched"
        self._spec.evaluation_metric = self._spec.accuracy

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        file_name = "train.tsv" if split == "train" else split + "_matched.tsv"
        return self.read_data_file(data_path, split, file_name, True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[8].strip()
        text2 = row[9].strip()
        label = row[11] if split == "train" else row[15] if split == "dev" else None
        return DataExample([text1, text2], label)


class GLUEMNLIMismatchedTask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("MNLI", "classification", 3, 2, "GLUE")
        self._spec.output_dir = "MNLI-Mismatched"
        self._spec.evaluation_metric = self._spec.accuracy

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        file_name = "train.tsv" if split == "train" else split + "_mismatched.tsv"
        return self.read_data_file(data_path, split, file_name, True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[8].strip()
        text2 = row[9].strip()
        label = row[11] if split == "train" else row[15] if split == "dev" else None
        return DataExample([text1, text2], label)


class GLUEQNLITask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("QNLI", "classification", 3, 2, "GLUE")
        self._spec.evaluation_metric = self._spec.accuracy

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, split + ".tsv", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[1].strip()
        text2 = row[2].strip()
        label = row[3] if split != "test" else None
        return DataExample([text1, text2], label)


class GLUEMRPCTask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("MRPC", "classification", 2, 2, "GLUE")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.binary_f1

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, "msr_paraphrase_" + split + ".txt", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[3].strip()
        text2 = row[4].strip()
        label = row[0]
        return DataExample([text1, text2], label)


class GLUERTETask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("RTE", "classification", 2, 2, "GLUE")
        self._spec.evaluation_metric = self._spec.accuracy

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, split + ".tsv", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[1].strip()
        text2 = row[2].strip()
        label = row[3] if split != "test" else None
        return DataExample([text1, text2], label)


class GLUESTSBTask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("STS-B", "regression", 1, 2, "GLUE")
        self._spec.evaluation_metric = self._spec.corr

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, split + ".tsv", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text1 = row[7].strip()
        text2 = row[8].strip()
        if split != "test":
            score = float(row[9]) - 1.0
            score = "%.5f" % (score / 4.0,)
        else:
            score = None
        return DataExample([text1, text2], score)

    def format_output(self, value: float):
        return "%.2f" % (1 + value * 4,)

    def postprocess_prediction(self, value: float):
        score = min(max(1 + value * 4, 0), 5)
        score = round(score)
        return (score - 1.0) / 4.0


class GLUESST2Task(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("SST-2", "classification", 2, 1, "GLUE")
        self._spec.evaluation_metric = self._spec.accuracy

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_data_file(data_path, split, split + ".tsv", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        text = row[0 if split != "test" else 1].strip()
        label = row[1] if split != "test" else None
        return DataExample(text, label)


class GLUEDiagnosticsTask(GLUETask):

    def __init__(self):
        self._spec = TaskSpecification("AX", "classification", 3, 2, "GLUE")
        self._spec.no_dev_set = True
        self._spec.evaluation_metric = self._spec.mcc

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        assert split in ("train", "test")
        if split == "train":
            return self.read_data_file(data_path, split, "../MNLI/train.tsv", True)
        elif split == "test":
            return self.read_data_file(data_path, split, "AX.tsv", True)

    def create_example(self, row: List[str], normalizer: TextNormalizer, split: str) -> DataExample:
        if split == "train":
            text1 = row[8].strip()
            text2 = row[9].strip()
            label = row[11]
        else:
            text1 = row[1].strip()
            text2 = row[2].strip()
            label = None
        return DataExample([text1, text2], label)


class PPCTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("PPC", "classification", 3, 2)
        self._spec.evaluation_metric = self._spec.accuracy

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        if split == "dev": split = "test"
        split_path = self.get_split_path(data_path, split, extension="jsonl")
        with open(split_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                obj = json.loads(line.strip())
                sent1, sent2, label = obj["sent1"], obj["sent2"], obj["label"]
                yield DataExample([sent1, sent2], label)


class UMOWYTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("UMOWY", "classification", 2, 1)
        self._spec.evaluation_metric = self._spec.binary_f1

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        return self.read_simple(data_path, split, separator="\t", label_first=True, normalize=False)


class UOKIKTask(BaseTask):

    def __init__(self):
        self._spec = TaskSpecification("UOKIK", "classification", 2, 1)
        self._spec.evaluation_metric = self._spec.binary_f1
        self._spec.no_dev_set = True

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        split_path = self.get_split_path(data_path, split, extension="csv")
        classes = {"KLAUZULA_ABUZYWNA": "1", "BEZPIECZNE_POSTANOWIENIE_UMOWNE": "0"}
        cl = [clazz for clazz, clazz_id in classes.items()]
        with open(split_path, "r", encoding="utf-8") as input_file:
            input_file.readline() # skip first line
            for line in input_file:
                line = line.strip()
                assert cl[0] in line or cl[1] in line, line
                for clazz, clazz_id in classes.items():
                    if clazz in line:
                        sent = line.replace(f"{clazz}", "")
                        sent = sent.strip("\",;„”").replace("\"\"", "\"")
                        label = clazz_id
                        yield DataExample(sent, label)



TASKS = {
    # Polish tasks
    "WCCRS_HOTELS":    WCCRSHotelsTask,
    "WCCRS_MEDICINE":  WCCRSMedicineTask,
    "SICK-E":          SICKEntailmentTask,
    "SICK-R":          SICKRelatednessTask,
    "8TAGS":           EightTagsTask,
    "KLEJ-NKJP":       KLEJNKJPTask,
    "KLEJ-CDS-E":      KLEJCDSEntailmentTask,
    "KLEJ-CDS-R":      KLEJCDSRelatednessTask,
    "KLEJ-CBD":        KLEJCBDTask,
    "KLEJ-POLEMO-IN":  KLEJPolEmoINTask,
    "KLEJ-POLEMO-OUT": KLEJPolEmoOUTTask,
    "KLEJ-DYK":        KLEJDYKTask,
    "KLEJ-PSC":        KLEJPSCTask,
    "KLEJ-ECR":        KLEJECRRegressionTask,
    "PPC":             PPCTask,
    "UMOWY":           UMOWYTask,

    # English tasks
    "GLUE-COLA":       GLUECoLATask,
    "GLUE-MNLI-MA":    GLUEMNLIMatchedTask,
    "GLUE-MNLI-MI":    GLUEMNLIMismatchedTask,
    "GLUE-QQP":        GLUEQQPTask,
    "GLUE-QNLI":       GLUEQNLITask,
    "GLUE-MRPC":       GLUEMRPCTask,
    "GLUE-RTE":        GLUERTETask,
    "GLUE-STS-B":      GLUESTSBTask,
    "GLUE-SST-2":      GLUESST2Task,
    "GLUE-AX":         GLUEDiagnosticsTask
}