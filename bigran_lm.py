# Now lets implement bigram languagage model
import torch
import torch.nn as nn 
from torch.nn import functional as F 

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly read
        # s off the logits for the nex token a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B, T) tensor of integers 
        logits = self.token_embedding_table(idx)  # (B, T, C)

        return logits
