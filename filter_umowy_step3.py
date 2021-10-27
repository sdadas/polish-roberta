import faiss
import torch
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VectorIndex(object):

    def __init__(self, records: List[Dict], create_empty: bool=False):
        self.create_empty = create_empty
        self.model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")
        self.index = self._create_index(records)
        self.records = records if not create_empty else []

    def _create_index(self, records: List[Dict]):
        sentences = [rec.get("sentence") for rec in records]
        embeddings = self.embed(sentences)
        dim = embeddings.shape[1]
        res = faiss.IndexFlatIP(dim)
        #res = faiss.IndexIVFFlat(quantizer, dim, 100)
        #res.train(embeddings)
        if not self.create_empty:
            res.add(embeddings)
        return res

    def embed(self, sentences: List[str]):
        embeddings = self.normalize_embeddings(self.model.encode(sentences, convert_to_tensor=True))
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def search(self, sentence: str, top_k=5):
        embeddings = self.embed([sentence])
        return self.search_vector(embeddings, top_k)

    def search_vector(self, vector, top_k=5):
        sim, indices = self.index.search(vector, top_k)
        sim = sim.flatten().tolist()
        indices = indices.flatten().tolist()
        indices = [ind for ind in indices if ind >= 0]
        res: List[Dict] = []
        for k, idx in enumerate(indices):
            record = self.records[idx]
            res.append({"sentence": record["sentence"], "label": record["label"], "sim": sim[k]})
        return res

    def add(self, sentence: str, label: str):
        embeddings = self.normalize_embeddings(self.model.encode([sentence], convert_to_tensor=True))
        self.index.add(embeddings.cpu().numpy())
        self.records.append({"sentence": sentence, "label": label})

    def normalize_embeddings(self, embeddings):
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class ClausesFilter(object):

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.exclude_pattern = re.compile(r"(\.\.|\d{2}-\d{3}|ciasteczk|kapitał zakładowy)", re.IGNORECASE)
        self.records: List[str] = self.__read_records()

    def run(self):
        print(len(self.records))
        filtered = self.__filter_with_faiss(self.records)
        with open("bezpieczne_klauzule_filtered.txt", "w", encoding="utf-8") as output_file:
            for record in filtered:
                sent = record.get("sentence")
                if '"' in sent or ',' in sent:
                    sent = sent.replace('"', '""')
                    sent = '"' + sent + '"'
                output_file.write(sent)
                output_file.write(",BEZPIECZNE_POSTANOWIENIE_UMOWNE")
                output_file.write("\n")

    def __read_records(self):
        res = set()
        with open(self.input_path, "r", encoding="utf -8") as input_file:
            for line in input_file:
                record = line.strip().split("\t")[1]
                record = record.replace('\\)', ')').strip(',:; ')
                record = re.sub("\s+", " ", record)
                if self.__is_upper_case(record):
                    record = record.lower()
                if record in res:
                    continue
                match = re.search(self.exclude_pattern, record)
                if match is not None:
                    continue
                res.add(record)
        return list(res)

    def __is_upper_case(self, record: str):
        upper = record.upper()
        equal = 0.0
        for idx, ch in enumerate(record):
            if ch == upper[idx]:
                equal += 1.0
        return (equal / len(record)) > 0.8

    def __filter_with_faiss(self, sentences: List[str]):
        records = [{"sentence": val, "label": "0"} for val in sentences]
        index = VectorIndex(records, create_empty=True)
        embeddings = index.embed(sentences)
        for idx, sent in enumerate(sentences):
            similar = index.search_vector(embeddings[idx:idx+1, :])
            sim = similar[0].get("sim") if len(similar) > 0 else 0.0
            if sim < 0.80:
                index.add(sent, "0")
            if idx % 1000 == 0: print(f"{idx}  {len(index.records)}")
        return index.records


if __name__ == '__main__':
    filter = ClausesFilter("bezpieczne_klauzule.txt")
    filter.run()
