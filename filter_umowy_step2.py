import json
from typing import List, Dict, TextIO

import torch
from sentence_transformers import SentenceTransformer
import itertools
import faiss

from tasks import UOKIKTask


class VectorIndex(object):

    def __init__(self, records: List[Dict]):
        self.model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
        self.index = self._create_index(records)
        self.records = records

    def _create_index(self, records: List[Dict]):
        sentences = [rec.get("sentence") for rec in records]
        embeddings = self.normalize_embeddings(self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False))
        dim = embeddings.shape[1]
        res = faiss.IndexFlatIP(dim)
        res.add(embeddings.cpu().numpy())
        return res

    def search(self, sentence: str, top_k=5):
        embeddings = self.normalize_embeddings(self.model.encode([sentence], convert_to_tensor=True, show_progress_bar=False))
        sim, indices = self.index.search(embeddings.cpu().numpy(), top_k)
        sim = sim.flatten().tolist()
        indices = indices.flatten().tolist()
        res: List[Dict] = []
        for k, idx in enumerate(indices):
            record = self.records[idx]
            res.append({"sentence": record["sentence"], "label": record["label"], "sim": sim[k]})
        return res

    def normalize_embeddings(self, embeddings):
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)



def index_example():
    task = UOKIKTask()
    iter = itertools.chain(task.read("data", "train"), task.read("data", "test"))
    records: List[Dict] = []
    for record in iter:
        records.append({"sentence": record.inputs[0], "label": record.label})
    index = VectorIndex(records)
    with open("bezpieczne_klauzule.txt", "w", encoding="utf-8") as output_file:
        __read_common_crawl("filtered.txt", output_file, index)

def __read_common_crawl(input_path: str, output_file: TextIO, index: VectorIndex):
    idx = 0
    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            json_value = json.loads(line.strip())
            sentences: List[str] = json_value.get("text")
            for sent in sentences:
                idx += 1
                __process_common_crawl_sentence(sent, output_file, index)
                if idx % 10000 == 0: print(idx)

def __process_common_crawl_sentence(sent: str, output_file: TextIO, index: VectorIndex):
    if len(sent) < 100: return
    lower = sent.lower()
    similar = index.search(sent)
    has_abusive = any([res for res in similar if res.get("label") == "1"])
    top_score = similar[0].get("sim")
    cookies = "cookies" in lower or "javascript" in lower
    if top_score > 0.7 and not has_abusive and not cookies:
        output_file.write(f"{top_score:.2f}\t{sent}\n")
        output_file.flush()


def __find_mismatched(record, results, output_file: TextIO, min_similarity=0.85):
    for res in results:
        label = res.get("label")
        if record.label != label and res.get("sim") > min_similarity:
            out = f"{record.inputs[0]},{record.label}\t{res.get('sentence')},{label}"
            output_file.write(out)
            output_file.write("\n")
            print(out)
            return

if __name__ == '__main__':
    res = index_example()
    print(res)