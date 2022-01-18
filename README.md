### Polish RoBERTa
This repository contains pre-trained [RoBERTa](https://arxiv.org/abs/1907.11692) models for Polish as well as evaluation code for several Polish linguistic tasks. The released models were trained using [Fairseq toolkit](https://github.com/pytorch/fairseq) in the National Information Processing Institute, Warsaw, Poland. We provide two models based on BERT base and BERT large architectures. Two versions of each model are available: one for [Fairseq](https://github.com/pytorch/fairseq) and one for [Huggingface Transformers](https://github.com/huggingface/transformers).

#### Updates

**18.01.2022** - We release the second version of the large model. This version has been trained using the same procedure as RoBERTa‑base-v2: unigram tokenizer, whole word masking, more update steps with lower batch size. We also utilised larger vocabulary of 128k entries.

**21.03.2021** - We release a new version of the base model. The updated model has been trained on the same corpus as the original model but we used different hyperparameters. We made the following changes: 1) Sentencepiece Unigram model was used instead of BPE, 2) The model was trained with whole-word masking objective instead of classic token masking, 3) We utilized the full context of 512 tokens so training examples could include more than one sentence (the original model was trained on single sencentes only), 4) Longer pretraining (400k steps).

#### Models

<table>
<thead>
<th>Model</th>
<th>L / H / A*</th>
<th>Batch size</th>
<th>Update steps</th>
<th>Corpus size</th>
<th>KLEJ Score**</th> 
<th>Fairseq</th>
<th>Transformers</th>
</thead>
<tr>
  <td>RoBERTa&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>125k</td>
  <td>~20GB</td>
  <td>85.39</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v3.4.0/roberta_base_transformers.zip">v3.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&#8209;v2&nbsp;(base)</td>
  <td>12&nbsp;/&nbsp;768&nbsp;/&nbsp;12</td>
  <td>8k</td>
  <td>400k</td>
  <td>~20GB</td>
  <td>86.72</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_fairseq.zip">v0.10.1</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_base_transformers.zip">v4.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>30k</td>
  <td>50k</td>
  <td>~135GB</td>
  <td>87.69</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_fairseq.zip">v0.9.0</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-transformers-v3.4.0/roberta_large_transformers.zip">v3.4</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&#8209;v2&nbsp;(large)</td>
  <td>24&nbsp;/&nbsp;1024&nbsp;/&nbsp;16</td>
  <td>2k</td>
  <td>400k</td>
  <td>~200GB</td>
  <td>88.87</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_fairseq.zip">v0.10.2</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_transformers.zip">v4.14</a>
  </td>
</tr>
</table>

\* L - the number of encoder blocks, H - hidden size, A - the number of attention heads <br/>
\** Average KLEJ score over 5 runs, see evaluation section for detailed results<br/>

More details are available in the paper [Pre-training Polish Transformer-based Language Models at Scale](https://arxiv.org/abs/2006.04229).

```
@InProceedings{dadas2020pretraining,
  title="Pre-training Polish Transformer-Based Language Models at Scale",
  author="Dadas, S{\l}awomir and Pere{\l}kiewicz, Micha{\l} and Po{\'{s}}wiata, Rafa{\l}",
  booktitle="Artificial Intelligence and Soft Computing",
  year="2020",
  publisher="Springer International Publishing",
  pages="301--314",
  isbn="978-3-030-61534-5"
}
```

### Getting started

#### How to use with Fairseq

```python
import os
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils

model_path = "roberta_large_fairseq"
loaded = hub_utils.from_pretrained(
    model_name_or_path=model_path,
    data_name_or_path=model_path,
    bpe="sentencepiece",
    sentencepiece_vocab=os.path.join(model_path, "sentencepiece.bpe.model"),
    load_checkpoint_heads=True,
    archive_map=RobertaModel.hub_models(),
    cpu=True
)
roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
roberta.eval()
input = roberta.encode("Zażółcić gęślą jaźń.")
output = roberta.extract_features(input)
print(output[0][1])
```

#### How to use with HuggingFace Transformers

```python
import torch, os
from transformers import RobertaModel, AutoModel, PreTrainedTokenizerFast

model_dir = "roberta_base_transformers"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model: RobertaModel = AutoModel.from_pretrained(model_dir)
input = tokenizer.encode("Zażółcić gęślą jaźń.")
output = model(torch.tensor([input]))[0]
print(output[0][1])
```

### Evaluation

To replicate our experiments, first download the required datasets using `download_data.py` script:

```bash
python download_data.py
```

Next, run `run_tasks.py` script to prepare the data, fine-tune and evaluate the model. We used the following parameters for each task:

```bash
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-NKJP --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-CDS-E --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-CDS-R --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 1 --tasks KLEJ-CBD --fp16 True --max-sentences 8 --update-freq 4 --resample 0:0.75,1:3
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-POLEMO-IN --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-POLEMO-OUT --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-DYK --fp16 True --max-sentences 8 --update-freq 4 --resample 0:1,1:3
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-PSC --fp16 True --max-sentences 8 --update-freq 4 --resample 0:1,1:3
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks KLEJ-ECR --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks 8TAGS --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks SICK-E --fp16 True --max-sentences 8 --update-freq 2
python run_tasks.py --arch roberta_base --model_dir roberta_base_fairseq --train-epochs 10 --tasks SICK-R --fp16 True --max-sentences 8 --update-freq 2
```

#### Evaluation results on KLEJ Benchmark
Below we show the evaluation results of our models on the tasks included in [KLEJ Benchmark](https://klejbenchmark.com/). We fine-tuned both models 5 times for each task.
Detailed scores for each run and averaged scores are presented in Table 1 and Table 2.

| Run     | NKJP | CDSC&#8209;E | CDSC&#8209;R | CBD   | PolEmo&#8209;IN | PolEmo&#8209;OUT | DYK   | PSC   | AR    | Avg     |
|---------|----------|--------|--------|-------|--------------|---------------|-------|-------|-------|---------|
| 1       |   93.15  |  93.30 |  94.26 | 66.67 |     91.97    |     78.74     | 66.86 | 98.63 | 87.75 |  **85.70**  |
| 2       |   93.93  |  94.20 |  93.94 | 68.16 |     91.83    |     75.91     | 65.93 | 98.77 | 87.93 |  **85.62**  |
| 3       |   94.22  |  94.20 |  94.04 | 69.23 |     90.17    |     76.92     | 65.69 | 99.24 | 87.76 |  **85.72**  |
| 4       |   93.97  |  94.70 |  93.98 | 63.81 |     90.44    |     76.32     | 65.18 | 99.39 | 87.58 |  **85.04**  |
| 5       |   93.63  |  94.00 |  93.96 | 65.95 |     90.58    |     74.09     | 65.92 | 98.48 | 87.08 |  **84.85**  |
| **Avg** |**93.78** | **94.08** |  **94.04** | **66.77** | **91.00** | **76.40** | **65.92** | **98.90** | **87.62** |  **85.39**  |

Table 1. KLEJ results for RoBERTa base model

| Run     | NKJP | CDSC&#8209;E | CDSC&#8209;R | CBD   | PolEmo&#8209;IN | PolEmo&#8209;OUT | DYK   | PSC   | AR    | Avg     |
|---------|----------|--------|--------|-------|--------------|---------------|-------|-------|-------|---------|
| 1       |   94.80  |  94.20 |  94.30 | 69.62 |     90.58    |     78.74     | 71.23 | 98.62 | 87.99 |  **86.68**  |
| 2       |   94.27  |  94.50 |  94.44 | 70.67 |     90.17    |     78.95     | 69.64 | 99.08 | 87.98 |  **86.63**  |
| 3       |   93.73  |  94.30 |  94.64 | 70.67 |     91.41    |     78.14     | 74.44 | 98.92 | 87.64 |  **87.10**  |
| 4       |   94.07  |  93.90 |  94.58 | 70.00 |     91.00    |     78.14     | 69.94 | 98.93 | 87.22 |  **86.42**  |
| 5       |   94.31  |  94.20 |  94.71 | 70.46 |     91.00    |     77.94     | 71.67 | 98.48 | 88.15 |  **86.77**  |
| **Avg** |**94.24** | **94.22** |  **94.54** | **70.28** | **90.83** | **78.38** | **71.38** | **98.81** | **87.80** |  **86.72**  |

Table 2. KLEJ results for RoBERTa-v2 base model

| Run     | NKJP | CDSC&#8209;E | CDSC&#8209;R | CBD   | PolEmo&#8209;IN | PolEmo&#8209;OUT | DYK   | PSC   | AR    | Avg     |
|---------|----------|--------|--------|-------|--------------|---------------|-------|-------|-------|---------|
| 1       |   94.31  |  93.50 |  94.63 | 72.39 |     92.80    |     80.54     | 71.87 | 98.63 | 88.82 |  **87.50**  |
| 2       |   95.14  |  93.90 |  94.93 | 69.82 |     92.80    |     82.59     | 73.39 | 98.94 | 88.96 |  **87.83**  |
| 3       |   95.24  |  93.30 |  94.61 | 71.59 |     91.41    |     82.19     | 75.35 | 98.64 | 89.31 |  **87.96**  |
| 4       |   94.46  |  93.20 |  94.96 | 71.08 |     92.80    |     82.39     | 70.59 | 99.09 | 88.60 |  **87.46**  |
| 5       |   94.46  |  93.00 |  94.82 | 69.83 |     92.11    |     83.00     | 74.85 | 98.79 | 88.65 |  **87.72**  |
| **Avg** |**94.72** | **93.38** |  **94.79** | **70.94** | **92.38** | **82.14** | **73.21** | **98.82** | **88.87** |  **87.69**  |

Table 3. KLEJ results for RoBERTa large model

| Run     | NKJP | CDSC&#8209;E | CDSC&#8209;R | CBD   | PolEmo&#8209;IN | PolEmo&#8209;OUT | DYK   | PSC   | AR    | Avg     |
|---------|----------|--------|--------|-------|--------------|---------------|-------|-------|-------|---------|
| 1       |   95.82  |  94.10 |  95.02 | 74.54 |     93.07    |     85.43     | 76.70 | 98.47 | 89.24 |  **89.15**  |
| 2       |   95.72  |  93.90 |  95.10 | 74.55 |     93.49    |     84.01     | 74.71 | 98.93 | 89.02 |  **88.83**  |
| 3       |   95.43  |  94.30 |  95.36 | 70.97 |     93.21    |     82.59     | 76.61 | 98.15 | 89.31 |  **88.44**  |
| 4       |   95.97  |  94.40 |  95.12 | 75.10 |     92.80    |     85.83     | 74.05 | 98.93 | 89.14 |  **89.04**  |
| 5       |   95.92  |  94.70 |  95.09 | 75.66 |     93.07    |     82.79     | 75.35 | 98.62 | 88.78 |  **88.89**  |
| **Avg** |**95.77** | **94.28** |  **95.14** | **74.16** | **93.13** | **84.13** | **75.48** | **98.62** | **89.10** |  **88.87**  |

Table 3. KLEJ results for RoBERTa-v2 large model

#### Evaluation results on other tasks

| Task                 | Task type                   | Metric |Base model (v1)                  | Large model (v1)                 |
|----------------------|-----------------------------|--------|-----------------------------|------------------------------|
| [SICK-E](https://github.com/sdadas/polish-sentence-evaluation/tree/master/resources/downstream) | Textual entailment     | Accuracy | 86.13    |  87.67|
| [SICK-R](https://github.com/sdadas/polish-sentence-evaluation/tree/master/resources/downstream) | Semantic relatedness        | Spearman correlation | 82.26    |  85.63  |
| [Poleval 2018 - NER](http://2018.poleval.pl/index.php/tasks/)  | Named&nbsp;entity&nbsp;recognition    | F1 score (exact match) | 87.94 | 89.98 |
| [8TAGS](https://github.com/sdadas/polish-sentence-evaluation/tree/master/resources/downstream) | Multi&nbsp;class&nbsp;classification  | Accuracy | 77.22 | 80.84 |
