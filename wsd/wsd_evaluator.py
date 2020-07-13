import operator
import os
import re
from collections import Counter
from typing import Dict, List, Any
import csv

from fairseq import hub_utils
from fairseq.models.roberta import RobertaHubInterface, RobertaModel
from scipy.special import softmax

from tasks import DataExample
from train.evaluator import TaskEvaluator
from utils.analyzer import PolishAnalyzer
from wsd.wsd_tasks import WordSenseTask, WordSensePolevalTask


class WordSenseEvalContext(object):

    def __init__(self):
        self.count = 0
        self.correct = 0
        self.top_errors = Counter()
        self.idx = 0


class WordSenseEvaluator(TaskEvaluator):

    def __init__(self, task: WordSenseTask, task_id: str, model: RobertaHubInterface, data_path: str, output_dir: str):
        super().__init__(task, task_id, model, data_path, output_dir)
        self._task = task
        self.dataset = task.get_dataset(data_path)
        self.analyzer = PolishAnalyzer()

    def wsd_evaluate(self):
        ctx: WordSenseEvalContext = WordSenseEvalContext()
        examples: List[Any] = self.dataset["examples"]
        results_path = f"{self._task.__class__.__name__}_results.csv"
        with open(results_path, "w", encoding="utf-8") as results_file:
            writer = csv.writer(results_file)
            for example in examples:
                self.wsd_evaluate_example(writer, ctx, example)

    def wsd_evaluate_example(self, writer: csv.writer, ctx: WordSenseEvalContext, example: Any):
        text: str = example["text"]
        senses: List[Any] = example["senses"]
        for sense in senses:
            start = sense["start"]
            end = sense["end"]
            text_modified = text[:start] + " ` " + text[start:end] + " ` " + text[end:]
            word = sense["word"]
            sense_id = sense["sense"]
            scores = self.wsd_predict(text_modified, word)
            senses_num = len(scores)
            scores["0"] = 0
            predicted_id = max(scores.items(), key=operator.itemgetter(1))[0]
            ctx.count += 1
            not_found = "" if sense_id in scores.keys() else "NOT_FOUND"
            matched = sense_id == predicted_id
            accuracy = "%.2f" % (ctx.correct * 100.0 / ctx.count,)
            row = [ctx.idx, word, "1" if matched else "0", not_found, senses_num, accuracy, predicted_id, sense_id, text]
            if sense_id == predicted_id: ctx.correct += 1
            else: ctx.top_errors[word] += 1
            writer.writerow(row)
            ctx.idx += 1
            if (ctx.idx % 1000) == 0:
                print(ctx.top_errors)

    def wsd_predict(self, sentence: str, lemma: str=None) -> Dict[str, float]:
        matched = re.search(r'`(.+)`', sentence, re.DOTALL)
        assert matched, "no word for disambiguation selected"
        word = matched.group(1).strip().lower()
        if not lemma:
            _, lemmas = self.analyzer.analyze(word)
            lemma = " ".join(lemmas)
        senses = self.dataset["words"].get(lemma, None)
        if not senses: return {}
        elif len(senses) == 1: return {senses[0]: 1.0}
        results = {}
        for sense in senses:
            score = self._predict_sense_score(sense, sentence)
            results[sense] = score
        return results

    def _predict_sense_score(self, sense_id: str, sentence: str) -> float:
        sense = self.dataset["senses"].get(sense_id, None)
        if not sense: return 0
        sentences = self._task.extract_sentences_from_sense(sense)
        if len(sentences) == 0: return 0
        records = [DataExample([sentence, sense_sentence], None) for sense_sentence in sentences]
        matched = 0.0
        for record in records:
            prediction = self.predict(record, logits=True)
            matched += softmax(prediction.cpu().detach().numpy()[0])[1]
        return matched / len(records)


if __name__ == '__main__':
    task = WordSensePolevalTask()
    model_dir = "roberta_large"
    output_dir: str = "data_processed"
    model_name: str = os.path.basename(model_dir)
    task_output_dir: str = os.path.join(output_dir, f"{task.spec().output_path()}-bin")
    checkpoints_output_dir = os.path.join("checkpoints", model_name, task.spec().output_path())
    checkpoint_file = "checkpoint_last.pt" if task.spec().no_dev_set else "checkpoint_best.pt"
    loaded = hub_utils.from_pretrained(
        model_name_or_path=checkpoints_output_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=task_output_dir,
        bpe="sentencepiece",
        sentencepiece_vocab=os.path.join(model_dir, "sentencepiece.bpe.model"),
        load_checkpoint_heads=True,
        archive_map=RobertaModel.hub_models()
    )
    roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
    evaluator = WordSenseEvaluator(task, "test", roberta, "data", output_dir)
    evaluator.wsd_evaluate()