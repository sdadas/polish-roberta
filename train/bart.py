import torch
from fairseq.models.bart import BARTHubInterface


class CustomBARTHubInterface(BARTHubInterface):

    def __init__(self, args, task, model):
        super().__init__(args, task, model)

    def encode(self, sentence: str, *addl_sentences, no_separator=True) -> torch.LongTensor:
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()

