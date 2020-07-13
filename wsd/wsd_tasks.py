import itertools
import json
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Iterable, Dict, Any, List, Set

from tasks import BaseTask, DataExample, TaskSpecification

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


class WordSenseTask(BaseTask):

    @abstractmethod
    def get_dataset(self, data_path: str):
        raise NotImplementedError

    @abstractmethod
    def extract_sentences_from_sense(self, sense: Dict[str, Any]):
        raise NotImplementedError


class WordSensePolevalTask(WordSenseTask):

    def __init__(self) -> None:
        self._spec = TaskSpecification("POLEVAL", "classification", 2, 2, "WSD")

    def quote(self):
        return " ` "

    def get_dataset(self, data_path: str):
        file_path = self.get_path(data_path)
        with open(file_path, "r", encoding="utf-8") as input_file:
            return json.load(input_file)

    def get_path(self, data_path: str) -> str:
        input_path = os.path.join(data_path, self.spec().task_path(), "data.json")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        return input_path

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        dataset = self.get_dataset(data_path)
        return self.generate_train(dataset) if split == "train" else self.generate_test(dataset)

    def generate_train(self, dataset: Dict[str, Any]):
        words = defaultdict(set)
        for key, values in dataset["words"].items():
            for val in values:
                words[val].add(key)
        for sense in words.keys():
            for example in self.generate_intra_sense_examples(dataset, sense):
                yield example
        processed: Set[str] = set()
        for key, values in dataset["words"].items():
            if len(values) < 2: continue
            for sense_id1, sense_id2 in itertools.combinations(values, 2):
                pair_key = "".join(sorted([sense_id1, sense_id2]))
                if pair_key in processed: continue
                for example in self.generate_inter_sense_examples(dataset, sense_id1, sense_id2):
                    yield example
                processed.add(pair_key)

    def generate_intra_sense_examples(self, dataset: Dict[str, Any], sense_id: str):
        sense = dataset["senses"].get(sense_id, None)
        if not sense: return []
        sentences = self.extract_sentences_from_sense(sense)
        if len(sentences) == 0: return []
        res = []
        for first, second in itertools.combinations(sentences, 2):
            res.append(DataExample([first, second], "1"))
        return res

    def generate_inter_sense_examples(self, dataset: Dict[str, Any], sense_id1: str, sense_id2: str):
        sense1 = dataset["senses"].get(sense_id1, None)
        sense2 = dataset["senses"].get(sense_id2, None)
        if not sense1 or not sense2: return []
        sentences1 = self.extract_sentences_from_sense(sense1)
        sentences2 = self.extract_sentences_from_sense(sense2)
        if len(sentences1) == 0 or len(sentences2) == 0: return []
        res = []
        for first in sentences1:
            for second in sentences2:
                res.append(DataExample([first, second], "0"))
        return res

    def generate_test(self, dataset: Dict[str, Any]):
        examples: List[Any] = dataset["examples"]
        for example in examples:
            text: str = example["text"]
            senses: List[Any] = example["senses"]
            for sense in senses:
                start = sense["start"]
                end = sense["end"]
                text_modified = text[:start] + self.quote() + text[start:end] + self.quote() + text[end:]
                word = sense["word"]
                sense_id = sense["sense"]
                sense_examples = self.generate_examples_for_text(dataset, text_modified, word, sense_id)
                for sense_example in sense_examples:
                    yield sense_example

    def generate_examples_for_text(self, dataset: Dict[str, Any], text: str, word: str, sense_id: str):
        senses = dataset["words"].get(word, None)
        if not senses:
            logging.warning("Word not found: %s", word)
            return []
        results: List[DataExample] = []
        for sense in senses:
            senses_list = dataset["senses"]
            sense_object = senses_list.get(sense, None)
            if not sense_object: continue
            label = "1" if sense == sense_id else "0"
            sentences = self.extract_sentences_from_sense(sense_object)
            for sentence in sentences:
                results.append(DataExample([text, sentence], label))
        return results

    def extract_sentences_from_sense(self, sense: Dict[str, Any]) -> List[str]:
        res = []
        definition = sense.get("def", None)
        sense_words = set()
        sense_words.update(sense.get("synonyms"))
        sense_words.update(sense.get("related"))
        if definition:
            definition = self.quote() + ", ".join(sense_words) + self.quote() + ":" + sense["def"].replace("**", "")
        elif len(sense_words) > 1:
            definition = self.quote() + ", ".join(sense_words) + self.quote()
        if definition: res.append(definition)
        sense_examples = sense.get("examples", [])
        for sense_example in sense_examples:
            res.append(sense_example.replace("**", self.quote()))
        return res
