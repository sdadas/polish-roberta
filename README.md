### Polish RoBERTa
This repository contains pre-trained [RoBERTa](https://arxiv.org/abs/1907.11692) models for Polish as well as evaluation code for several Polish linguistic tasks. The released models were trained using [Fairseq toolkit](https://github.com/pytorch/fairseq) in the National Information Processing Institute, Warsaw, Poland. We provide two models based on BERT base and BERT large architectures. Two versions of each model are available: one for [Fairseq](https://github.com/pytorch/fairseq) and one for [Huggingface Transformers](https://github.com/huggingface/transformers).

<table>
<thead>
<th>Model</th>
<th>Encoder blocks</th>
<th>Batch size</th>
<th>Update steps</th>
<th>Corpus size</th>
<th>Final perplexity*</th>
<th>Fairseq</th>
<th>Transformers</th>
</thead>
<tr>
  <td>RoBERTa&nbsp;(base)</td>
  <td>16</td>
  <td>8k</td>
  <td>125k</td>
  <td>~20GB</td>
  <td>3.66</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip">Download</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_transformers.zip">Download</a>
  </td>
</tr>
<tr>
  <td>RoBERTa&nbsp;(large)</td>
  <td>24</td>
  <td>30k</td>
  <td>50k</td>
  <td>~135GB</td>
  <td>2.92</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_fairseq.zip">Download</a>
  </td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_transformers.zip">Download</a>
  </td>
</tr>
</table>

\* Perplexity of the best checkpoint, computed on the validation split

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
import torch
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import RobertaModel, AutoModel

model_dir = "roberta_large_transformers"
tokenizer = SentencePieceBPETokenizer(f"{model_dir}/vocab.json", f"{model_dir}/merges.txt")
getattr(tokenizer, "_tokenizer").post_processor = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))
model: RobertaModel = AutoModel.from_pretrained(model_dir)

input = tokenizer.encode("Zażółcić gęślą jaźń.")
output = model(torch.tensor([input.ids]))[0]
print(output[0][1])
```