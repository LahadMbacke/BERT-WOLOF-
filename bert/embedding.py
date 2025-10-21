import torch
import math
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=128):
        super().__init__()

        pe = torch.zeros(max_len,d_model) # 
        pe.requires_grad = False

        for pos in range(max_len):
            for i in range(d_model// 2):
                pe[pos,2*i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos,2*i + 1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
        
        self.pe = pe.unsqueeze(0) # shape (1, max_len, d_model)

    def forward(self,x):
        return self.pe
    

class BertEmbedding(nn.Module):
    """Bert Embedding Layer combining Token, Positional and Segment Embeddings
        -Token Embedding: Converts token indices to dense vectors
        -Positional Embedding: Adds positional information to the token embeddings
        -Segment Embedding: Differentiates between segments in tasks like QA
        Args:
            vocab_size (int): Size of the vocabulary
            embed_size (int): Dimension of the embedding vectors
            seq_len (int): Maximum sequence length
            drpout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(self,vocab_size,embed_size,seq_len=64,drpout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token = nn.Embedding(vocab_size,embed_size)
        self.segment = nn.Embedding(3,embed_size)
        self.position = PositionalEmbedding(d_model=embed_size,max_len=seq_len)
        self.dropout = nn.Dropout(drpout)

    def forward(self,x,segment_ids):
        x = self.token(x) + self.position(x).to(x.device) + self.segment(segment_ids)
        return self.dropout(x)