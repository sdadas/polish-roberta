import itertools
import json
import logging
import os
from collections import defaultdict
from random import choice
from typing import Iterable, Dict, Any, List, Set, Union

from tasks import BaseTask, DataExample, TaskSpecification

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


class WordSensePolevalTask(BaseTask):

    def __init__(self) -> None:
        self._spec = TaskSpecification("POLEVAL", "classification", 2, 2, "WSD")

    def quote(self):
        return " ` "

    def get_path(self, data_path: str) -> str:
        input_path = os.path.join(data_path, self.spec().task_path(), "data.json")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        return input_path

    def read(self, data_path: str, split: str) -> Iterable[DataExample]:
        file_path = self.get_path(data_path)
        with open(file_path, "r", encoding="utf-8") as input_file:
            dataset = json.load(input_file)
        return self.generate_train(dataset) if split == "train" else self.generate_test(dataset)

    def generate_train(self, dataset: Dict[str, Any]):
        words = defaultdict(set)
        for key, values in dataset["words"].items():
            for val in values:
                words[val].add(key)
        for sense in words.keys():
            for example in self.generate_intra_sense_examples(dataset, sense, words):
                yield example
        processed: Set[str] = set()
        for key, values in dataset["words"].items():
            if len(values) < 2: continue
            for sense_id1, sense_id2 in itertools.combinations(values, 2):
                pair_key = "".join(sorted([sense_id1, sense_id2]))
                if pair_key in processed: continue
                for example in self.generate_inter_sense_examples(dataset, sense_id1, sense_id2, words):
                    yield example
                processed.add(pair_key)

    def generate_intra_sense_examples(self, dataset: Dict[str, Any], sense_id: str, words: Dict[str, Set]):
        sense = dataset["senses"].get(sense_id, None)
        if not sense: return []
        sense_words: Set[str] = words.get(sense_id)
        sentences = self._extract_sentences_from_sense(sense, sense_words)
        if len(sentences) == 0: return []
        res = []
        for first, second in itertools.combinations(sentences, 2):
            res.append(DataExample([first, second], "1"))
        return res

    def generate_inter_sense_examples(self, dataset: Dict[str, Any], sense_id1: str, sense_id2: str, words: Dict[str, Set]):
        sense1 = dataset["senses"].get(sense_id1, None)
        sense2 = dataset["senses"].get(sense_id2, None)
        if not sense1 or not sense2: return []
        sense_words1: Set[str] = words.get(sense_id1)
        sense_words2: Set[str] = words.get(sense_id2)
        sentences1 = self._extract_sentences_from_sense(sense1, sense_words1)
        sentences2 = self._extract_sentences_from_sense(sense2, sense_words2)
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
            sentences = self._extract_sentences_from_sense(sense_object, word)
            for sentence in sentences:
                results.append(DataExample([text, sentence], label))
        return results

    def _extract_sentences_from_sense(self, sense: Dict[str, Any], sense_words: Union[str,Set]) -> List[str]:
        res = []
        definition = sense.get("def", None)
        if definition:
            word = sense_words if isinstance(sense_words, str) else choice(tuple(sense_words))
            definition = self.quote() + word + self.quote() + ":" + sense["def"].replace("**", "")
            res.append(definition)
        sense_examples = sense.get("examples", [])
        for sense_example in sense_examples:
            res.append(sense_example.replace("**", self.quote()))
        return res
