import operator
import os
import re
from typing import Dict, List, Any

from fairseq import hub_utils
from fairseq.models.roberta import RobertaHubInterface, RobertaModel
from scipy.special import softmax

from tasks import DataExample
from train.evaluator import TaskEvaluator
from utils.analyzer import PolishAnalyzer
from wsd.wsd_tasks import WordSenseTask, WordSensePolevalTask


class WordSenseEvaluator(TaskEvaluator):

    def __init__(self, task: WordSenseTask, task_id: str, model: RobertaHubInterface, data_path: str, output_dir: str):
        super().__init__(task, task_id, model, data_path, output_dir)
        self._task = task
        self.dataset = task.get_dataset(data_path)
        self.analyzer = PolishAnalyzer()

    def wsd_evaluate(self):
        count = 0
        correct = 0
        examples: List[Any] = self.dataset["examples"]
        for example in examples:
            text: str = example["text"]
            senses: List[Any] = example["senses"]
            for sense in senses:
                start = sense["start"]
                end = sense["end"]
                text_modified = text[:start] + " ` " + text[start:end] + " ` " + text[end:]
                word = sense["word"]
                sense_id = sense["sense"]
                scores = self.wsd_predict(text_modified, word)
                scores["0"] = 0
                predicted_sense_id = max(scores.items(), key=operator.itemgetter(1))[0]
                count += 1
                not_found = "" if sense_id in scores.keys() else "NOT_FOUND"
                if sense_id == predicted_sense_id:
                    correct += 1
                    print("YES %s %s %.2f" % (word, not_found, correct / count,))
                else:
                    print("-   %s %s %.2f" % (word, not_found, correct / count,))

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
            score = self._predict_sense_score(sense, sentence, lemma)
            results[sense] = score
        return results

    def _predict_sense_score(self, sense_id: str, sentence: str, lemma: str) -> float:
        sense = self.dataset["senses"].get(sense_id, None)
        if not sense: return 0
        sentences = self._task.extract_sentences_from_sense(sense, lemma)
        records = [DataExample([sentence, sense_sentence], None) for sense_sentence in sentences]
        matched = 0.0
        for record in records:
            prediction = self.predict(record, logits=True)
            #matched += (1.0 if prediction == "1" else 0.0)
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