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
<th>Download</th>
</thead>
<tr>
  <td>RoBERTa (base)</td>
  <td>16</td>
  <td>8k</td>
  <td>125k</td>
  <td>~20GB</td>
  <td>3.66</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_fairseq.zip">Fairseq</a>&nbsp;
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_base_transformers.zip">Transformers</a>
  </td>
</tr>
<tr>
  <td>RoBERTa (large)</td>
  <td>24</td>
  <td>30k</td>
  <td>50k</td>
  <td>~135GB</td>
  <td>2.92</td>
  <td>
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_fairseq.zip">Fairseq</a>&nbsp;
  <a href="https://github.com/sdadas/polish-roberta/releases/download/models/roberta_large_transformers.zip">Transformers</a>
  </td>
</tr>
</table>

\* Perplexity of the best checkpoint, computed on the validation split