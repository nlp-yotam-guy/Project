from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from collections import namedtuple
from torch.autograd import Variable
from torch import optim
import random
import numpy as np
from Pytorch_Decoder import *
from Pytorch_Encoder import *

use_cuda = torch.cuda.is_available() # To check if GPU is available
MAX_LENGTH = 20 # We restrict our experiments to sentences of length 20 or less
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



def check_and_convert_to_cuda(var):
    return var.cuda() if use_cuda else var


def construct_vocab(sentences):
    word2id = dict()
    id2word = dict()
    word2id[SOS_TOKEN] = 0
    word2id[EOS_TOKEN] = 1
    id2word[0] = SOS_TOKEN
    id2word[1] = EOS_TOKEN
    for sentence in sentences:
        for word in sentence.strip().split(' '):
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(word2id)-1] = word
    return Vocabulary(word2id, id2word)


def sent_to_word_id(sentences, vocab, eos=True, is_eval=0):
    data = []
    words = sentences
    if is_eval == 1:
        if eos:
            end = [vocab.word2id[EOS_TOKEN]]
        else:
            end = []
        words = words.strip().split(' ')
        while '' in words:
            words.remove('')
        print(len(words))
        if len(words) < MAX_LENGTH:
            print("words in eval are:", words)
            data.append([vocab.word2id[w] for w in words] + end)
            print("data is :", data)
    else:
        for sent in sentences:
            if eos:
                end = [vocab.word2id[EOS_TOKEN]]
            else:
                end = []
            # if is_eval == 1:
            #     print("in eval sent ", sent)
            words = sent.strip().split(' ')
            # if is_eval == 1:
            #     print("in eval words ", words)
            if len(words) < MAX_LENGTH:
                # if is_eval == 1:
                #     for w in words:
                #         print("in eval ", w)
                data.append([vocab.word2id[w] for w in words] + end)
    return data

def init_weights(m):
    if not hasattr(m, 'weight'):
        return
    if type(m) == nn.Conv1d:
        width = m.weight.data.shape[-1] / (m.weight.data.shape[0] ** 0.5)
    else:
        width = 0.05

    m.weight.data.uniform_(-width, width)


def trainIters(encoder_a, encoder_c, decoder, n_iters, batch_size=32, learning_rate=1e-4, print_every=100):
    encoder_a_optimizer = optim.Adam(encoder_a.parameters(), lr=learning_rate)
    encoder_c_optimizer = optim.Adam(encoder_c.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Sample a training pair
    training_pairs = list(zip(*(normal_dataset, simple_dataset)))

    criterion = nn.NLLLoss()

    print_loss_total = 0

    # The important part of the code is the 3rd line, which performs one training
    # step on the batch. We are using a variable `print_loss_total` to monitor
    # the loss value as the training progresses

    for itr in range(1, n_iters + 1):
        training_pair = random.sample(training_pairs, k=batch_size)
        input_variable, target_variable = list(zip(*training_pair))

        loss = train(input_variable, target_variable, encoder_a, encoder_c,
                     decoder, encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer,
                     criterion, batch_size=batch_size)

        print_loss_total += loss

        if itr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print(print_loss_avg)
            print_loss_total = 0
    print("Training Completed")


def train(input_variables, output_variables, encoder_a, encoder_c, decoder,
          encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, batch_size=32):
    # Initialize the gradients to zero
    encoder_a_optimizer.zero_grad()
    encoder_c_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    for count in range(batch_size):
        # Length of input and output sentences
        input_variable = input_variables[count]
        output_variable = output_variables[count]

        input_length = input_variable.size()[0]
        output_length = output_variable.size()[0]

        loss = 0

        # Encoder outputs: We use this variable to collect the outputs
        # from encoder after each time step. This will be sent to the decoder.
        position_ids = Variable(torch.LongTensor(range(0, input_length)))
        position_ids = position_ids.cuda() if use_cuda else position_ids
        cnn_a = encoder_a(position_ids, input_variable)
        cnn_c = encoder_c(position_ids, input_variable)

        cnn_a = cnn_a.cuda() if use_cuda else cnn_a
        cnn_c = cnn_c.cuda() if use_cuda else cnn_c

        prev_word = Variable(torch.LongTensor([[0]]))  # SOS
        prev_word = prev_word.cuda() if use_cuda else prev_word

        decoder_hidden = decoder.initHidden()

        for i in range(output_length):
            decoder_output, decoder_hidden = \
                decoder(prev_word, decoder_hidden, cnn_a, cnn_c)
            topv, topi = decoder_output.data.topk(1)
            # Maybe .item???
            ni = topi[0][0]
            prev_word = Variable(torch.LongTensor([[ni]]))
            prev_word = prev_word.cuda() if use_cuda else prev_word
            loss += criterion(decoder_output, output_variable[i].unsqueeze(0))

            if ni == 1:  # EOS
                break

    # Backpropagation
    loss.backward()
    encoder_a_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item() / output_length


def evaluate(sent_pair, encoder_a, encoder_c, decoder, source_vocab, target_vocab, max_length=MAX_LENGTH):
    source_sent = sent_to_word_id(sent_pair[0], source_vocab, is_eval=1)
    print("my src sent is: ", source_sent)
    if len(source_sent) == 0:
        return
    source_sent = source_sent[0]
    input_variable = Variable(torch.LongTensor(source_sent))

    if use_cuda:
        input_variable = input_variable.cuda()

    input_length = input_variable.size()[0]
    position_ids = Variable(torch.LongTensor(range(0, input_length)))
    position_ids = position_ids.cuda() if use_cuda else position_ids
    cnn_a = encoder_a(position_ids, input_variable)
    cnn_c = encoder_c(position_ids, input_variable)
    cnn_a = cnn_a.cuda() if use_cuda else cnn_a
    cnn_c = cnn_c.cuda() if use_cuda else cnn_c
    prev_word = Variable(torch.LongTensor([[0]]))  # SOS
    prev_word = prev_word.cuda() if use_cuda else prev_word
    decoder_hidden = decoder.initHidden()
    target_sent = []
    ni = 0
    out_length = 0
    while not ni == 1 and out_length < 10:
        decoder_output, decoder_hidden = \
            decoder(prev_word, decoder_hidden, cnn_a, cnn_c)

        topv, topi = decoder_output.data.topk(1)
        print("test1: ", topv, topi)
        ni = topi[0][0].item()
        print("test2: ",ni)
        target_sent.append(target_vocab.id2word[ni])
        prev_word = Variable(torch.LongTensor([[ni]]))
        print("test3")
        prev_word = prev_word.cuda() if use_cuda else prev_word
        out_length += 1

    print("Source: " + sent_pair[0])
    print("Translated: " + ' '.join(target_sent))
    print("Expected: " + sent_pair[1])
    print("")

f_normal = open("../data/wiki/normal.aligned", 'r', encoding="utf8")
f_simple = open("../data/wiki/simple.aligned", 'r', encoding="utf8")

#f_normal = open("C:\\Users\\guyazov\\Desktop\\sentence-aligned.v2\\normal.aligned", 'r', encoding="utf8")
normal_data_raw = f_normal.readlines()
#f_simple = open("C:\\Users\\guyazov\\Desktop\\sentence-aligned.v2\\simple.aligned", 'r', encoding="utf8")
simple_data_raw = f_simple.readlines()
normal_data_sample = normal_data_raw[:100]
simple_data_sample = simple_data_raw[:100]
Vocabulary = namedtuple('Vocabulary', ['word2id', 'id2word']) # A Named tuple representing the vocabulary of a particular language
normal_vocab = construct_vocab(normal_data_raw)
simple_vocab = construct_vocab(simple_data_raw)
normal_data = sent_to_word_id(normal_data_raw, normal_vocab)
simple_data = sent_to_word_id(simple_data_raw, simple_vocab)
normal_dataset = [Variable(torch.LongTensor(sent)) for sent in normal_data]
simple_dataset = [Variable(torch.LongTensor(sent)) for sent in simple_data]
if use_cuda: # And if cuda is available use the cuda tensor types
    normal_dataset = [i.cuda() for i in normal_dataset]
    simple_dataset = [i.cuda() for i in simple_dataset]
encoder_a = ConvEncoder(len(normal_vocab.word2id), embedding_size, dropout=dropout,
                        num_channels_attn=attn_units, num_channels_conv=conv_units,
                        num_layers=encoder_layers)
encoder_c = ConvEncoder(len(normal_vocab.word2id), embedding_size, dropout=dropout,
                        num_channels_attn=attn_units, num_channels_conv=conv_units,
                        num_layers=encoder_layers)
decoder = AttnDecoder(len(simple_vocab.word2id), dropout=dropout,
                      hidden_size_gru=hidden_size_gru, embedding_size=embedding_size,
                      attn_size=attn_units, cnn_size=conv_units)

if use_cuda:
    encoder_a = encoder_a.cuda()
    encoder_c = encoder_c.cuda()
    decoder = decoder.cuda()

encoder_a.apply(init_weights)
encoder_c.apply(init_weights)
decoder.apply(init_weights)

encoder_a.training = True
encoder_c.training = True
decoder.training = True

trainIters(encoder_a, encoder_c, decoder, num_iterations, print_every=print_every, batch_size=batch_size)
encoder_a.training = False
encoder_c.training = False
decoder.training = False

for i in range(len(normal_data_sample)):
    evaluate((normal_data_sample[i], simple_data_sample[i]), encoder_a, encoder_c, decoder, normal_vocab, simple_vocab)
