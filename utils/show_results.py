import os
import json
from collections import defaultdict
from typing import Dict, List

from utils.table import TablePrinter, TableColumn

SCORES = ("accuracy", "binary-f1", "spearman", "1-wmae")

def parse_runlog(path: str):
    results = defaultdict(lambda: defaultdict(list))
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            model_dir = obj["params"]["model_dir"]
            model_results = results[model_dir]
            parse_runlog_line(obj, model_results)
    for key, val in results.items():
        print(key)
        print_table(val)
        print("\n")

def parse_runlog_line(obj: Dict[str, any], results: Dict[str, List]):
    scores = obj["scores"]
    found = None
    for check_score in SCORES:
        found = scores.get(check_score, None)
        if found is not None:
            break
    task = obj["task"]
    results[task].append(found)

def print_table(results: Dict[str, List]):
    header = sorted([key for key in results.keys()])
    table = [header]
    max_scores = max([len(val) for val in results.values()])
    for row_idx in range(max_scores):
        row = []
        for key in header:
            scores = results.get(key)
            value = ""
            if len(scores) > row_idx:
                value = "%.2f" % (scores[row_idx] * 100,)
            row.append(value)
        table.append(row)
    means = []
    for key in header:
        scores = results.get(key)
        means.append("%.2f" % ((sum(scores) / len(scores)) * 100,))
    table.append(["-----" for _ in range(len(header) + 1)])
    table.append(means)
    header.insert(0, "MEAN")
    for row in table[1:]:
        if row[0] == "-----": continue
        row_mean = sum(float(val) for val in row if len(val) > 0) / len([row for val in row if len(val) > 0])
        row.insert(0, "%.2f" % (row_mean,))
    printer: TablePrinter = TablePrinter()
    for idx in range(0, len(header)): printer.column(idx, align=TableColumn.ALIGN_CENTER, width=15)
    printer.print(table)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    runlog_path = os.path.join(script_dir, os.path.pardir, "runlog.txt")
    parse_runlog(runlog_path)