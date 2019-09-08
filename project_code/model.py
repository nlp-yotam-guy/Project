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
    def __init__(self, vocab_size, embedding_matrix, embedding_size, max_len, dropout=0.2,
                 num_channels_attn=512, num_channels_conv=512,
                 kernel_size=3, num_layers=5):
        super(ConvEncoder, self).__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_embedding.weight.data.copy_(embedding_matrix)
        self.word_embedding.weight.requires_grad = False
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

    def __init__(self, output_vocab_size, word_freq, use_cuda, dropout=0.2, hidden_size_lstm=128,
                 cnn_size=128, attn_size=128, n_layers_lstm=4,
                 embedding_size=128):

        super(AttnDecoder, self).__init__()

        self.n_lstm_layers = n_layers_lstm
        self.hidden_size_lstm = hidden_size_lstm
        self.output_vocab_size = output_vocab_size
        self.dropout = dropout
        self.word_freq = word_freq
        self.word_count = dict()

        self.embedding = nn.Embedding(output_vocab_size, hidden_size_lstm)
        self.lstm = nn.LSTM(hidden_size_lstm + embedding_size, hidden_size_lstm,
                            n_layers_lstm, bidirectional=True)
        self.transform_lstm_hidden_in = nn.Linear(hidden_size_lstm, embedding_size)
        self.transform_lstm_hidden_out = nn.Linear(2*hidden_size_lstm*n_layers_lstm, embedding_size)
        self.dense_o = nn.Linear(hidden_size_lstm, output_vocab_size)

        self.n_layers_lstm = n_layers_lstm

        self.use_cuda = use_cuda

    def forward(self, y_i, g_i, h_i, cnn_a, cnn_c, input_sentence, pos, vocab_simple):
        g_i = self.embedding(y_i)
        g_i = F.dropout(g_i, self.dropout, self.training)
        d_i = self.transform_lstm_hidden_in(h_i) + g_i
        s_i = torch.bmm(d_i, cnn_a)
        s_i = s_i.view(1, -1)
        a_i = F.softmax(s_i)

        c_i = torch.bmm(a_i.view(1, 1, -1), cnn_c.transpose(1, 2))
        lstm_output, lstm_h = self.lstm(torch.cat((x, c_i), dim=-1))
        lstm_h = lstm_h[0].flatten(0, -1)
        lstm_h = lstm_h.reshape((1,1,lstm_h.size()[0]))
        lstm_h = self.transform_lstm_hidden_out(lstm_h)
        lstm_hidden = F.dropout(lstm_h[0], self.dropout, self.training)
        softmax_output = F.log_softmax(self.dense_o(lstm_hidden))

        return softmax_output, lstm_hidden


    # function to initialize the hidden layer of LSTM.
    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers_lstm, 1, self.hidden_size_lstm))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class Rephraser:

    def __init__(self,embed_dim, max_len, drop_prob,
                 hidden_size, batch_size, n_epoches, vocab_normal, vocab_simple,
                 vocab_size_normal, vocab_size_simple, word_freq, n_conv_layers,
                 learning_rate, use_cuda, kernel_size=3, embedding_matrix=None,teacher_forcing_ratio = 0.5):

        self.embed_dim = embed_dim
        self.max_len = max_len
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.vocab_normal = vocab_normal
        self.vocab_simple = vocab_simple
        self.vocab_size_normal = vocab_size_normal
        self.vocab_size_simple = vocab_size_simple
        self.word_freq = word_freq
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.use_cuda = use_cuda
        self.lr = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.loss_graph = []

        try:
            self.embedding_matrix_normal = torch.from_numpy(embedding_matrix[0])
            self.embedding_matrix_simple = torch.from_numpy(embedding_matrix[1])

        except:
            self.embedding_matrix_normal = torch.from_numpy(np.zeros((self.vocab_size_normal,self.embed_dim),
                                                                     dtype=np.float32))
            self.embedding_matrix_simple = torch.from_numpy(np.zeros((self.vocab_size_simple, self.embed_dim),
                                                                     dtype=np.float32))

        self.embedding_matrix_normal = self.embedding_matrix_normal.cuda() if self.use_cuda \
            else self.embedding_matrix_normal
        self.embedding_matrix_simple = self.embedding_matrix_simple.cuda() if self.use_cuda \
            else self.embedding_matrix_simple

        self.encoder_a = None
        self.encoder_c = None
        self.decoder = None
        self.criterion = None
        self.encoder_a_optimizer = None
        self.encoder_c_optimizer = None
        self.decoder_optimizer = None

        self.define()

    def init_weights(self,m):

        if not hasattr(m, 'weight'):
            return
        if type(m) == nn.Conv1d:
            width = m.weight.data.shape[-1] / (m.weight.data.shape[0] ** 0.5)
        else:
            width = 0.05

        m.weight.data.uniform_(-width, width)

    def define(self):
        self.encoder_a = ConvEncoder(self.vocab_size_normal, self.embedding_matrix_normal, self.embed_dim, self.max_len, dropout=self.drop_prob,
                                num_channels_attn=self.hidden_size, num_channels_conv=self.hidden_size,
                                num_layers=3*self.n_conv_layers)
        self.encoder_c = ConvEncoder(self.vocab_size_normal, self.embedding_matrix_normal, self.embed_dim, self.max_len, dropout=self.drop_prob,
                                num_channels_attn=self.hidden_size, num_channels_conv=self.hidden_size,
                                num_layers=self.n_conv_layers)
        self.decoder = AttnDecoder(self.vocab_size_simple, self.word_freq, self.use_cuda, dropout=self.drop_prob,
                              hidden_size_lstm=self.hidden_size, embedding_size=self.embed_dim,
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


    def get_initial_encoding(self):
        decoder_output = np.random.normal(0, 1, (1, self.vocab_size_simple)).astype(np.float32)
        decoder_output = torch.from_numpy(decoder_output)
        decoder_output = torch.softmax(decoder_output, -1)
        decoder_output = decoder_output.cuda() if self.use_cuda else decoder_output
        decoder_output = torch.mm(decoder_output, self.embedding_matrix_simple)
        return decoder_output


    def save_model(self, iter, loss):
        torch.save({
            'iter': iter,
            'encoder_a_state_dict': self.encoder_a.state_dict(),
            'encoder_c_state_dict': self.encoder_c.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_a_optimizer': self.encoder_a_optimizer.state_dict(),
            'encoder_c_optimizer': self.encoder_c_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'loss': loss,
            'loss_graph': self.loss_graph
            }, 'saved_model_weights')

    def trainIters(self, input_dataset, output_dataset, print_every=100):

        # Sample a training pair
        training_pairs = [(input_dataset[i],output_dataset[i]) for i in range(len(input_dataset))]
        idx = list(range(len(training_pairs)))

        for itr in range(1, self.n_epoches + 1):
            random.shuffle(idx)
            training_pair = self.create_batch(training_pairs,idx)
            # for instance - training pair[0] is a tokenized sentence
            #  => [107, 655,  68, 106,  11, 656, 455, 657, 158,   1]

            input_variable, target_variable = list(zip(*training_pair))

            # # uncomment to print sentences:
            # k=10
            # for i in range(k):
            #     print([self.vocab_normal.id2word[j.item()] for j in input_variable[i]])
            #     print([self.vocab_simple.id2word[j.item()] for j in target_variable[i]],'\n')

            loss = self.train(input_variable, target_variable)
            self.loss_graph.append(loss)

            if itr % print_every == 0:
                print(itr, loss)
                self.save_model(itr,loss)

        print("Training Completed")

    def train(self, input_variables, output_variables):

        # Initialize the gradients to zero
        self.encoder_a_optimizer.zero_grad()
        self.encoder_c_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        for count in range(self.batch_size):
            # Length of input and output sentences
            input_variable = input_variables[count]
            output_variable = output_variables[count]

            # # uncomment to print sentences:
            # a = [input_variable[k].item() for k in range(len(input_variable))]
            # print([self.vocab_normal.id2word[a[i]] for i in range(len(input_variable))])
            # b = [output_variable[k].item() for k in range(len(output_variable))]
            # print([self.vocab_simple.id2word[b[i]] for i in range(len(output_variable))],'\n')

            input_length = input_variable.size()[0]
            output_length = output_variable.size()[0]

            loss = 0

            position_ids = Variable(torch.LongTensor(list(range(0, input_length))))
            position_ids = position_ids.cuda() if self.use_cuda else position_ids

            cnn_a = self.encoder_a(position_ids, input_variable)
            cnn_c = self.encoder_c(position_ids, input_variable)

            cnn_a = cnn_a.cuda() if self.use_cuda else cnn_a
            cnn_c = cnn_c.cuda() if self.use_cuda else cnn_c

            prev_word = Variable(torch.LongTensor([[0]]))  # SOS
            prev_word = prev_word.cuda() if self.use_cuda else prev_word

            # to feed the RNN step with weighted sum of the embedding matrix
            decoder_output = self.get_initial_encoding()
            decoder_hidden = self.decoder.initHidden()

            for i in range(output_length):
                decoder_hidden = decoder_hidden[-1].view(1,1,-1)
                decoder_output, decoder_hidden = \
                    self.decoder(prev_word, decoder_output, decoder_hidden, cnn_a, cnn_c, input_variable, i,
                                 self.vocab_simple)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0].item()

                if use_teacher_forcing:
                    prev_word = Variable(torch.LongTensor([[output_variable[i]]]))
                else:
                    prev_word = Variable(torch.LongTensor([[ni]]))
                prev_word = prev_word.cuda() if self.use_cuda else prev_word
                # one_hot = self.to_one_hot(output_variable[i])
                out = Variable(torch.LongTensor([output_variable[i]]))
                out = out.cuda() if self.use_cuda else out
                loss += self.criterion(decoder_output, out))

                if ni == 1:  # EOS
                    break

        # Backpropagation
        loss.backward()
        self.encoder_a_optimizer.step()
        self.encoder_c_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / output_length


    # evaluate the the model
    def evaluate(self, sent_pair):
        self.encoder_a.training = False
        self.encoder_c.training = False
        self.decoder.training = False
        source_sent = sent_pair[0]
        if (len(source_sent) == 0):
            return
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

        decoder_output = self.get_initial_encoding()
        decoder_hidden = self.decoder.initHidden()
        target_sent = []
        ni = 0
        out_length = 0
        while not ni == 1 and out_length < self.max_len:
            decoder_hidden = decoder_hidden[-1].view(1, 1, -1)
            decoder_output, decoder_hidden = \
                self.decoder(prev_word, decoder_output, decoder_hidden, cnn_a, cnn_c, input_variable, out_length, self.vocab_simple)

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0].item()
            target_sent.append(self.vocab_simple.id2word[ni])
            prev_word = Variable(torch.LongTensor([[ni]]))
            prev_word = prev_word.cuda() if self.use_cuda else prev_word
            out_length += 1

        orig_sent = word_id_to_sent(sent_pair[0], self.vocab_normal)
        expected_sent = word_id_to_sent(sent_pair[1], self.vocab_simple)
        print("Source: " + orig_sent)
        print("Translated: " + ' '.join(target_sent))
        print("Expected: " + expected_sent)
        print("")









