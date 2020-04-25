import multiprocessing
from typing import List, TextIO, Dict
from functools import partial

from sentencepiece import SentencePieceProcessor


def spm_initializer(spm_path: str):
    global spm
    spm = SentencePieceProcessor()
    spm.load(spm_path)
    global vocab
    vocab = {index:spm.id_to_piece(index) for index in range(spm.GetPieceSize())}

def spm_process(doc: List[str], max_length: int=None):
    return [" ".join([vocab.get(val) for val in encode_tokens(sent, max_length)]) for sent in doc]

def spm_process_ids(doc: List[str], max_length: int=None):
    return [" ".join([str(val) for val in encode_tokens(sent, max_length)]) for sent in doc]

def encode_tokens(sent: str, max_length: int=None):
    res = [val for val in spm.encode_as_ids(sent)]
    if max_length is not None:
        res = res[0:max_length]
    return res

def write_docs(outfile: TextIO, docs: List[List[str]], newline: bool):
    for doc in docs:
        for sentence in doc:
            outfile.write(sentence)
            outfile.write("\n")
        if newline: outfile.write("\n")

def get_doc(infile: TextIO):
    res = []
    for line in infile:
        if not line.strip():
            yield res
            res = []
        res.append(line)
    yield res

def process_input(outfile: TextIO, infile: TextIO, spm_path: str, threads: int, encode_ids: bool, newline: bool, max_length):
    process_func = spm_process_ids if encode_ids else spm_process
    process_func = partial(process_func, max_length=max_length)
    batch = []
    pool = multiprocessing.Pool(processes=threads, initializer=spm_initializer, initargs=(spm_path,))
    for doc in get_doc(infile):
        if len(batch) >= 1000:
            batch = [encoded for encoded in pool.imap(process_func, batch)]
            write_docs(outfile, batch, newline)
            batch = []
        batch.append(doc)
    if len(batch) > 0:
        batch = [encoded for encoded in pool.imap(process_func, batch)]
        write_docs(outfile, batch, newline)

def spm_encode(path: str, output_path: str, spm_path: str, threads: int=8, encode_ids: bool=True,
               newline: bool=False, max_length: int=None):
    with open(output_path, "w", encoding="utf-8") as outfile:
        with open(path, "r", encoding="utf-8") as infile:
            process_input(outfile, infile, spm_path, threads, encode_ids, newline, max_length)
