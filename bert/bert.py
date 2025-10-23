from torch import nn
from bert.embedding import BertEmbedding
from bert.encoder_layer import EncoderLayer


class BertModel(nn.Module):

    def __init__(self,vocab_size,d_model = 768,n_layers = 12,num_heads = 12, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.hidden_dim = 4 * d_model
        self.dropout = dropout

        # Embedding for tokens, positions and segments
        self.embedding = BertEmbedding(vocab_size = vocab_size, embed_size = d_model)

        # multiple Encoder layers 
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayer(
                    d_model = d_model,
                    num_heads = num_heads,
                    hidden_dim = self.hidden_dim,
                    dropout = dropout
                ) for _ in range(n_layers)
            ]
        )

    def forward(self, x, segment_info):
        """
        Forward pass of the BERT model.

        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len)
            segment_info: Tensor that indicates which sentence each token belongs to (A or B)
        Returns:
            Tensor of contextualized token embeddings with shape (batch_size, seq_len, d_model)
        """

        # Create an attention mask to ignore padding tokens (value = 0)
        # (x > 0) returns True for real tokens, False for padding.
        # After reshaping, the mask has shape (batch_size, 1, 1, seq_len)
        # so it can be broadcast correctly in multi-head attention.
        attention_mask = (x > 0).unsqueeze(1).unsqueeze(2)

        # Convert token IDs into embeddings (token + position + segment embeddings)
        # Output shape: (batch_size, seq_len, d_model)
        x = self.embedding(x, segment_info)

        # Pass the embeddings through each encoder layer
        # Each layer applies: 
        #   - multi-head self-attention (with mask)
        #   - feed-forward network
        #   - layer normalization and residual connections
        for encoder in self.encoder_blocks:
            x = encoder(x, attention_mask)

        # Return the final hidden states
        # These are contextualized embeddings where each token now "knows" about others.
        return x

class NextSentencePrediction(nn.Module):
    """
        Next Sentence Prediction Head for BERT
        class classification model : is_next, is_not_next
    """
    def __init__(self,hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim,2)

    def forward(self,x):
        # Return raw logits for loss computation (do not apply softmax here)
        x = self.linear(x[:,0])  # Use the embedding of the [CLS] token
        return x
    


class MaskedLanguageModeling(nn.Module):
    """
        Masked Language Modeling Head for BERT
    """
    def __init__(self,hidden_dim,vocab_size):
        """
        Args:
            hidden_dim : output dimension from BERT model
            vocab_size : size of the vocabulary for prediction
        """
        super().__init__()
        self.linear = nn.Linear(hidden_dim,vocab_size)

    def forward(self,x):
        # Return raw logits for each token position
        x = self.linear(x)  # Predict token for each position
        return x
    

class BertForPreTraining(nn.Module):
    """
        BERT model with heads for Pre-Training tasks:
            - Masked Language Modeling (MLM)
            - Next Sentence Prediction (NSP)
    """
    def __init__(self,bert:BertModel,vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence_prediction = NextSentencePrediction(bert.d_model)
        self.masked_language_modeling = MaskedLanguageModeling(bert.d_model,vocab_size)

    def forward(self,x,segment_info):
       x = self.bert(x,segment_info)
       return self.next_sentence_prediction(x), self.masked_language_modeling(x)
  