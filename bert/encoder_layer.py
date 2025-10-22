from torch import nn
from bert.attention import MultiHeadSelfAttention
from bert.feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, token_embeddings, attention_mask=None):
        """
        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, d_model)
                → The input embeddings for each token in the sequence.
            attention_mask: Optional tensor of shape (batch_size, 1, 1, seq_len)
                → Mask to prevent attending to certain positions (e.g. padding tokens).

        Returns:
            encoded_output: Tensor of shape (batch_size, seq_len, d_model)
                → The output representation after attention and feed-forward processing.
        """

        # ---- Self-Attention sublayer ----
        # Each token attends to other tokens in the sequence
        attention_output = self.self_attention(
            query=token_embeddings,
            key=token_embeddings,
            value=token_embeddings,
            mask=attention_mask
        )

        # Residual connection + dropout
        attention_residual = self.dropout(attention_output + token_embeddings)

        # Layer normalization for stable training
        normalized_attention = self.layer_norm(attention_residual)

        # ---- Feed-Forward sublayer ----
        # Non-linear transformation applied independently to each token
        feedforward_output = self.feed_forward(normalized_attention)

        # Residual connection + dropout again
        feedforward_residual = self.dropout(feedforward_output + normalized_attention)

        return self.layer_norm(feedforward_residual)
