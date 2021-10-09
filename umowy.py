import random
from fairseq import data

def read_file(file_path: str, label: str):
    res = []
    with open(file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            res.append(label + "\t" + line.strip())
    return res


if __name__ == '__main__':
    positvie = read_file("data/UMOWY/uokik.txt", "1")
    negative = read_file("data/UMOWY/exported.txt", "0")
    random.shuffle(negative)
    negative = negative[:10000]
    res = []
    res.extend(positvie)
    res.extend(negative)
    random.shuffle(res)

    train = res[:12284]
    dev = res[12284:13284]
    test = res[13284:]
    with open("data/UMOWY/train.txt", "w", encoding="utf-8") as output_file:
        for record in train:
            output_file.write(record)
            output_file.write("\n")
    with open("data/UMOWY/dev.txt", "w", encoding="utf-8") as output_file:
        for record in dev:
            output_file.write(record)
            output_file.write("\n")
    with open("data/UMOWY/test.txt", "w", encoding="utf-8") as output_file:
        for record in test:
            output_file.write(record)
            output_file.write("\n")