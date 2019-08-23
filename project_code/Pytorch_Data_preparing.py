from __future__ import unicode_literals, print_function, division
from io import open
from collections import namedtuple
import torch
from torch.autograd import Variable


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


def sent_to_word_id(sentences, vocab, eos=True):
    data = []
    for sent in sentences:
        if eos:
            end = [vocab.word2id[EOS_TOKEN]]
        else:
            end = []
        words = sent.strip().split(' ')

        if len(words) < MAX_LENGTH:
            data.append([vocab.word2id[w] for w in words] + end)
    return data


f = open("C:\\Users\\guyazov\\Desktop\\sentence-aligned.v2\\simple.aligned", 'r', encoding="utf8")
data = f.readlines()
Vocabulary = namedtuple('Vocabulary', ['word2id', 'id2word']) # A Named tuple representing the vocabulary of a particular language
simple_vocab = construct_vocab(data)
simple_data = sent_to_word_id(data, simple_vocab)
simple_dataset = [Variable(torch.LongTensor(sent)) for sent in simple_data]
#output_dataset = [Variable(torch.LongTensor(sent)) for sent in english_data]
if use_cuda: # And if cuda is available use the cuda tensor types
    input_dataset = [i.cuda() for i in input_dataset]
    output_dataset = [i.cuda() for i in output_dataset]