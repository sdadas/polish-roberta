import sys

import torch
from fairseq import checkpoint_utils

if __name__ == '__main__':
    checkpoint_path = sys.argv[1]
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
    del state["last_optimizer_state"]
    output_path = checkpoint_path + ".reduced"
    with open(output_path, "wb") as output_file:
        torch.save(state, output_file)