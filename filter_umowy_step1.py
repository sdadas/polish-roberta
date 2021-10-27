import logging
from typing import Dict, List

from tqdm import tqdm

from tasks import UMOWYTask, DataExample
from train.evaluator import TaskEvaluatorBuilder, TaskEvaluator
import json

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.root.setLevel(logging.DEBUG)


class CCProcessor(object):

    def __init__(self, data_path: str, model_dir: str):
        self.data_path = data_path
        self.model_dir = model_dir
        self.batch_size = 8
        self.model: TaskEvaluator = self._load_model()

    def _load_model(self) -> TaskEvaluator:
        builder = TaskEvaluatorBuilder(UMOWYTask(), "roberta_base", self.model_dir)
        return builder.build()

    def filter_data(self):
        with open(self.data_path, "r", encoding="utf-8") as infile, open("filtered.txt", "w", encoding="utf-8") as outfile:
            batch_text = []
            batch_lines = []
            for idx, line in enumerate(tqdm(infile)):
                batch_lines.append(line)
                batch_text.append(self._text(line.strip()))
                if len(batch_lines) >= self.batch_size:
                    accept = self._should_accept(batch_text)
                    for rec_idx, pred in enumerate(accept):
                        if pred == "1":
                            outfile.write(batch_lines[rec_idx])
                    batch_text = []
                    batch_lines = []

    def _text(self, json_str: str):
        json_val: Dict = json.loads(json_str)
        text = " ".join(json_val.get("text"))
        if len(text) > 2000:
            text = text[:2000]
        return text

    def _should_accept(self, batch: List[str]):
        records = [DataExample(text, None) for text in batch]
        return self.model.predict_batch(records)


if __name__ == '__main__':
    proc = CCProcessor("/Datasets/CommonCrawl/data.txt", "roberta_base_fairseq")
    proc.filter_data()