from process_data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import random


PRINT_PROGRESS = 1
MAX_EVAL_PRINT = 15


class ConvEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len, dropout=0.2,
                 num_channels_attn=512, num_channels_conv=512,
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

class AttnDecoder(nn.Module):

    def __init__(self, output_vocab_size, use_cuda, dropout=0.2, hidden_size_gru=128,
                 cnn_size=128, attn_size=128, n_layers_gru=1,
                 embedding_size=128):

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

        self.use_cuda = use_cuda

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
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class Rephraser:

    def __init__(self,embed_dim, max_len, drop_prob,
                 hidden_size, batch_size, n_epoches, vocab,
                 vocab_size, n_conv_layers, learning_rate, use_cuda, kernel_size=3, embedding_matrix=None):

        self.embed_dim = embed_dim
        self.embedding_matrix = embedding_matrix
        self.max_len = max_len
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.use_cuda = use_cuda
        self.lr = learning_rate

        if embedding_matrix is None:
            self.embedding_matrix = np.zeros((self.vocab_size,self.embed_dim))

        self.encoder_a = None
        self.encoder_c = None
        self.decoder = None
        self.criterion = None
        self.encoder_a_optimizer = None
        self.encoder_c_optimizer = None
        self.decoder_optimizer = None

        self.define()

        # print(self.model.summary())

    def init_weights(self,m):

        if not hasattr(m, 'weight'):
            return
        if type(m) == nn.Conv1d:
            width = m.weight.data.shape[-1] / (m.weight.data.shape[0] ** 0.5)
        else:
            width = 0.05

        m.weight.data.uniform_(-width, width)

    def define(self):
        self.encoder_a = ConvEncoder(self.vocab_size, self.embed_dim, self.max_len, dropout=self.drop_prob,
                                num_channels_attn=self.hidden_size, num_channels_conv=self.hidden_size,
                                num_layers=self.n_conv_layers)
        self.encoder_c = ConvEncoder(self.vocab_size, self.embed_dim, self.max_len, dropout=self.drop_prob,
                                num_channels_attn=self.hidden_size, num_channels_conv=self.hidden_size,
                                num_layers=self.n_conv_layers)
        self.decoder = AttnDecoder(self.vocab_size, self.max_len, dropout=self.drop_prob,
                              hidden_size_gru=self.hidden_size, embedding_size=self.embed_dim,
                              attn_size=self.hidden_size, cnn_size=self.hidden_size)

        if self.use_cuda:
            self.encoder_a = self.encoder_a.cuda()
            self.encoder_c = self.encoder_c.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder_a.apply(self.init_weights)
        self.encoder_c.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        self.encoder_a.training = True
        self.encoder_c.training = True
        self.decoder.training = True

        self.encoder_a_optimizer = optim.Adam(self.encoder_a.parameters(), lr=self.lr)
        self.encoder_c_optimizer = optim.Adam(self.encoder_c.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.encoder_a_optimizer.zero_grad()
        self.encoder_c_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        self.criterion = nn.NLLLoss()

    def create_batch(self,training_pairs,idx):
        batch = []
        j=0
        for i in idx:
            batch.append(training_pairs[i])
            j += 1
            if j > self.batch_size:
                break
        return batch

    def trainIters(self, input_dataset, output_dataset, print_every=100):

        # Sample a training pair
        # training_pairs = list(zip(*(input_dataset, output_dataset)))

        training_pairs = [(input_dataset[i],output_dataset[i]) for i in range(len(input_dataset))]
        idx = list(range(len(training_pairs)))


        # k = 10
        # for i in range(k):
        #     print([self.vocab.id2word[j] for j in training_pairs[i][0]])
        #     print([self.vocab.id2word[j] for j in training_pairs[i][1]], '\n')

        print_loss_total = 0

        # The important part of the code is the 3rd line, which performs one training
        # step on the batch. We are using a variable `print_loss_total` to monitor
        # the loss value as the training progresses

        for itr in range(1, self.n_epoches + 1):
            random.shuffle(idx)
            training_pair = self.create_batch(training_pairs,idx)

            # k=10
            # for i in range(k):
            #     print([self.vocab.id2word[j] for j in training_pair[i][0]])
            #     print([self.vocab.id2word[j] for j in training_pair[i][1]],'\n')

            input_variable, target_variable = list(zip(*training_pair))

            # k=10
            # for i in range(k):
            #     print([self.vocab.id2word[j] for j in input_variable[i]])
            #     print([self.vocab.id2word[j] for j in target_variable[i]],'\n')

            loss = self.train(input_variable, target_variable, self.encoder_a_optimizer, self.encoder_c_optimizer,
                              self.decoder_optimizer)

            print_loss_total += loss

            if itr % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print(print_loss_avg)
                print_loss_total = 0
        print("Training Completed")

    def train(self,input_variables, output_variables,
              encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer,):

        # Initialize the gradients to zero
        self.encoder_a_optimizer.zero_grad()
        self.encoder_c_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        for count in range(self.batch_size):
            # Length of input and output sentences
            input_variable = input_variables[count]
            output_variable = output_variables[count]

            # input_length = input_variable.size()[0]
            # output_length = output_variable.size()[0]

            input_length = len(input_variable)
            output_length = len(output_variable)

            loss = 0

            # Encoder outputs: We use this variable to collect the outputs
            # from encoder after each time step. This will be sent to the decoder.
            position_ids = Variable(torch.LongTensor(range(0, input_length)))
            position_ids = position_ids.cuda() if self.use_cuda else position_ids
            cnn_a = self.encoder_a(position_ids, input_variable)
            cnn_c = self.encoder_c(position_ids, input_variable)

            cnn_a = cnn_a.cuda() if self.use_cuda else cnn_a
            cnn_c = cnn_c.cuda() if self.use_cuda else cnn_c

            prev_word = Variable(torch.LongTensor([[0]]))  # SOS
            prev_word = prev_word.cuda() if self.use_cuda else prev_word

            decoder_hidden = self.decoder.initHidden()

            for i in range(output_length):
                decoder_output, decoder_hidden = \
                    self.decoder(prev_word, decoder_hidden, cnn_a, cnn_c)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                prev_word = Variable(torch.LongTensor([[ni]]))
                prev_word = prev_word.cuda() if self.use_cuda else prev_word
                loss += self.criterion(decoder_output, output_variable[i])

                if ni == 1:  # EOS
                    break

        # Backpropagation
        loss.backward()
        encoder_a_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / output_length


    # evaluate the the model
    def evaluate(self,sent_pair, vocab):
        self.encoder_a.training = False
        self.encoder_c.training = False
        self.decoder.training = False
        source_sent = sent_to_word_id(np.array([sent_pair[0]]), vocab)
        if (len(source_sent) == 0):
            return
        source_sent = source_sent[0]
        input_variable = Variable(torch.LongTensor(source_sent))

        if self.use_cuda:
            input_variable = input_variable.cuda()

        input_length = input_variable.size()[0]
        position_ids = Variable(torch.LongTensor(range(0, input_length)))
        position_ids = position_ids.cuda() if self.use_cuda else position_ids
        cnn_a = self.encoder_a(position_ids, input_variable)
        cnn_c = self.encoder_c(position_ids, input_variable)
        cnn_a = cnn_a.cuda() if self.use_cuda else cnn_a
        cnn_c = cnn_c.cuda() if self.use_cuda else cnn_c

        prev_word = Variable(torch.LongTensor([[0]]))  # SOS
        prev_word = prev_word.cuda() if self.use_cuda else prev_word

        decoder_hidden = self.decoder.initHidden()
        target_sent = []
        ni = 0
        out_length = 0
        while not ni == 1 and out_length < 10:
            decoder_output, decoder_hidden = \
                self.decoder(prev_word, decoder_hidden, cnn_a, cnn_c)

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            target_sent.append(vocab.id2word[ni])
            prev_word = Variable(torch.LongTensor([[ni]]))
            prev_word = prev_word.cuda() if self.use_cuda else prev_word
            out_length += 1

        print("Source: " + sent_pair[0])
        print("Translated: " + ' '.join(target_sent))
        print("Expected: " + sent_pair[1])
        print("")









