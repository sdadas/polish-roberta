import json
import os
from collections import defaultdict
from math import floor
from typing import Dict, List

from run_tasks import TASKS

SCORES = ("accuracy", "binary-f1", "spearman", "1-wmae")
TASK_NAME_MAPPING = {
    "KLEJ-NKJP": {"name": "nkjp-ner", "target": "target"},
    "KLEJ-CDS-E": {"name": "cdsc-e", "target": "entailment_judgment"},
    "KLEJ-CDS-R": {"name": "cdsc-r", "target": "relatedness_score"},
    "KLEJ-CBD": {"name": "cbd", "target": "target"},
    "KLEJ-DYK": {"name": "dyk", "target": "target"},
    "KLEJ-PSC": {"name": "psc", "target": "label"},
    "KLEJ-POLEMO-IN":  {"name": "polemo2.0-in", "target": "target"},
    "KLEJ-POLEMO-OUT": {"name": "polemo2.0-out", "target": "target"},
    "KLEJ-ECR": {"name": "ar", "target": "rating"}
}

def parse_runlog(path: str, output_path: str):
    results = defaultdict(lambda: defaultdict(list))
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            model_dir = obj["params"]["model_dir"]
            model_results = results[model_dir]
            parse_runlog_line(obj, model_results)
    for model_name in results.keys():
        export_model(model_name, results[model_name], output_path)

def parse_runlog_line(obj: Dict[str, any], results: Dict[str, List]):
    scores = obj["scores"]
    found = None
    for check_score in SCORES:
        found = scores.get(check_score, None)
        if found is not None:
            break
    task = obj["task"]
    result = {"id": obj["id"], "score": found, "task": task}
    results[task].append(result)

def export_model(model_name: str, results: Dict[str, any], output_path: str):
    min_scores = min([len(results[key]) for key in results.keys()])
    for idx in range(min_scores):
        scores = []
        for key in results.keys():
            score = results[key][idx]
            scores.append(score)
        export_model_run(f"{model_name}.{idx}", scores, output_path)
    median_scores = []
    for key in results.keys():
        scores = results[key]
        scores = sorted(scores, key=lambda v: v["score"])
        median_idx = floor(len(scores) / 2.0)
        median_scores.append(scores[median_idx])
    export_model_run(f"{model_name}.median", median_scores, output_path)

def export_model_run(run_name: str, scores: List[any], output_path: str):
    run_dir = os.path.join(output_path, run_name)
    os.makedirs(run_dir, exist_ok=True)
    meta = {}
    for score in scores:
        export_model_run_task(run_dir, score)
        meta[score["task"]] = score["score"]
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2, sort_keys=True)

def export_model_run_task(run_dir: str, score: Dict[str, any]):
    task_name = score["task"]
    task = TASKS[task_name]()
    filename = score["id"] + ".txt"
    pred_path = None
    script_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoints_dir = os.path.join(script_dir, os.path.pardir, "checkpoints")
    for root, dirs, files in os.walk(checkpoints_dir):
        if filename in files and task.spec().output_dir in root:
            pred_path = os.path.join(root, filename)
            break
    assert pred_path is not None
    task_info = TASK_NAME_MAPPING.get(task_name, None)
    if task_info is None: return
    with open(pred_path, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
    lines.insert(0, f"{task_info['target']}\n")
    output_filename = f"test_pred_{task_info['name']}.tsv"
    output_path = os.path.join(run_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for line in lines:
            output_file.write(line)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    runlog_path = os.path.join(script_dir, os.path.pardir, "runlog.txt")
    output_path = os.path.join(script_dir, os.path.pardir, "klej_results")
    parse_runlog(runlog_path, output_path)