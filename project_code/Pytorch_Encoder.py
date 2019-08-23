from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available() # To check if GPU is available
MAX_LENGTH = 20 # We restrict our experiments to sentences of length 10 or less
embedding_size = 256
hidden_size_gru = 256
attn_units = 256
conv_units = 256
num_iterations = 75000
print_every = 100
batch_size = 1
sample_size = 1000
dropout = 0.2
encoder_layers = 3
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


class ConvEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.2,
                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,
                 kernel_size=3, num_layers=5):
        super(ConvEncoder, self).__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size,
                                             padding=kernel_size // 2) for _ in range(num_layers)])

    def forward(self, position_ids, sentence_as_wordids):
        # Retrieving position and word embeddings
        position_embedding = self.position_embedding(position_ids)
        word_embedding = self.word_embedding(sentence_as_wordids)

        # Applying dropout to the sum of position + word embeddings
        embedded = F.dropout(position_embedding + word_embedding, self.dropout, self.training)

        # Transform the input to be compatible for Conv1d as follows
        # Length * Channel ==> Num Batches * Channel * Length
        embedded = torch.unsqueeze(embedded.transpose(0, 1), 0)

        # Successive application of convolution layers followed by residual connection
        # and non-linearity

        cnn = embedded
        for i, layer in enumerate(self.conv):
            # layer(cnn) is the convolution operation on the input cnn after which
            # we add the original input creating a residual connection
            cnn = F.tanh(layer(cnn) + cnn)

        return cnn
