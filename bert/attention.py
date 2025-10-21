import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model,num_heads,dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads # Dimension of each head
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)


    def forward(self, query,key,value, mask=None):
        """ query,key,value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, 1, 1, seq_len) 
            returns: shape (batch_size, seq_len, d_model)
        """

        # (batch_size, seq_len, d_model) 
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, d_model) -> (batch_size, max_len, num_heads, d_k) -> (batch_size, num_heads, max_len, d_k)
        # query = query.view(query.shape[0], -1, self.num_heads, self.d_k).permut(0, 2,1,3)
        # key = key.view(key.shape[0], -1, self.num_heads, self.d_k).permute(0, 2,1,3)
        # value = value.view(value.shape[0], -1, self.num_heads, self.d_k).permute(0, 2,1,3)

        # we split into num_heads first then transpose for attention dot product:
        query = query.view(query.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            # (batch_size, num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn, value)
        # (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        # we transpose and then reshape to combine all heads
        output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.num_heads * self.d_k)
        output = self.out(output)
        return output



# ==== TEST ====

batch_size = 2     # nombre d'exemples dans le batch
seq_len = 4        # longueur de la séquence (ex: nombre de mots)
d_model = 8        # dimension du vecteur d'entrée
num_heads = 2      # nombre de têtes d'attention

# Création d’un tenseur d’entrée factice
x = torch.randn(batch_size, seq_len, d_model)  # (2, 4, 8)

# Initialisation du module
attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

# Passage avant (forward)
output = attn(x, x, x)

print("✅ Entrée :", x.shape)
print("✅ Sortie :", output.shape)
print("🧠 Exemple de sortie :\n", output)









