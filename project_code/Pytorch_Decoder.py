from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
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

class AttnDecoder(nn.Module):

    def __init__(self, output_vocab_size, dropout=0.2, hidden_size_gru=128,
                 cnn_size=128, attn_size=128, n_layers_gru=1,
                 embedding_size=128, max_sentece_len=MAX_LENGTH):

        super(AttnDecoder, self).__init__()

        self.n_gru_layers = n_layers_gru
        self.hidden_size_gru = hidden_size_gru
        self.output_vocab_size = output_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(output_vocab_size, hidden_size_gru)
        self.gru = nn.GRU(hidden_size_gru + embedding_size, hidden_size_gru,
                          n_layers_gru)
        self.transform_gru_hidden = nn.Linear(hidden_size_gru, embedding_size)
        self.dense_o = nn.Linear(hidden_size_gru, output_vocab_size)

        self.n_layers_gru = n_layers_gru

    def forward(self, y_i, h_i, cnn_a, cnn_c):

        g_i = self.embedding(y_i)
        g_i = F.dropout(g_i, self.dropout, self.training)

        d_i = self.transform_gru_hidden(h_i) + g_i
        a_i = F.softmax(torch.bmm(d_i, cnn_a).view(1, -1))

        c_i = torch.bmm(a_i.view(1, 1, -1), cnn_c.transpose(1, 2))
        gru_output, gru_hidden = self.gru(torch.cat((g_i, c_i), dim=-1), h_i)

        gru_hidden = F.dropout(gru_hidden, self.dropout, self.training)
        softmax_output = F.log_softmax(self.dense_o(gru_hidden[-1]))

        return softmax_output, gru_hidden

    # function to initialize the hidden layer of GRU.
    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers_gru, 1, self.hidden_size_gru))
        if use_cuda:
            return result.cuda()
        else:
            return result
