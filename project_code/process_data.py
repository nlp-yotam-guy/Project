import numpy as np
import torch
from torch.autograd import Variable
import re
import unicodedata

WIKI_SIMPLE = '/normal.aligned'
WIKI_NORMAL = '/simple.aligned'
NEWSELA = '/newsela_articles_20150302.aligned.sents.txt'

SOS_TOKEN = "<sos>"
EOS_TOKEN = "."

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def load_wiki(wiki_normal, wiki_simple, limit_sent_len=-1, limit_data=-1):
    f_wiki_simple = open(wiki_simple, 'r', encoding="utf8")
    f_wiki_normal = open(wiki_normal, 'r', encoding="utf8")

    normal_sents_orig = f_wiki_normal.readlines()
    simple_sents_orig = f_wiki_simple.readlines()

    normal_sents = []
    simple_sents = []

    if limit_data == -1:
        limit_data = len(normal_sents_orig)

    i = 0
    for normal_line, simple_line in zip(normal_sents_orig, simple_sents_orig):
        if i > limit_data:
            break
        normal_sent = normalizeString(normal_line.split('\t')[-1])
        simple_sent = normalizeString(simple_line.split('\t')[-1])

        if normal_sent == simple_sents:
            continue

        normal_sent = normal_sent.split(' ')

        simple_sent = simple_sent.split(' ')
        del normal_sent[-1]
        normal_sent.append('.')
        del simple_sent[-1]
        simple_sent.append('.')

        normal_sents.append(normal_sent)
        simple_sents.append(simple_sent)
        i += 1

    return normal_sents, simple_sents


def load_newsela(newsela, limit_sent_len=-1, limit_data=-1):
    normal_sents = []
    simple_sents = []
    i = 0

    if limit_data == -1:
        limit_data = float('inf')
    if limit_sent_len == -1:
        limit_sent_len = float('inf')

    f = open(newsela, 'r')
    for line in f:
        if i > limit_data:
            break
        splited_line = line.split('\t')
        if splited_line[-2] == splited_line[-1]:
            continue
        normal_sent = splited_line[-2]
        normal_sent = normalizeString(normal_sent).split(' ')

        if normal_sent[-1] != '.':
            normal_sent.append('.')
        simple_sent = splited_line[-1]
        simple_sent = normalizeString(simple_sent).split(' ')
        simple_sent[-1] = simple_sent[-1].replace('\n', '')

        normal_sents.append(normal_sent)
        simple_sents.append(simple_sent)
        i += 1

    return normal_sents, simple_sents


def load_data(base_path, dataset, limit_sent=-1, limit_data=-1):
    normal_sents = simple_sents = None
    if dataset == 'wiki':
        wiki_normal = base_path + dataset + WIKI_NORMAL
        wiki_simple = base_path + dataset + WIKI_SIMPLE
        normal_sents, simple_sents = load_wiki(wiki_normal, wiki_simple, limit_sent, limit_data)

    elif dataset == 'newsela':
        newsela = base_path + dataset + NEWSELA
        normal_sents, simple_sents = load_newsela(newsela, limit_sent, limit_data)

    return normal_sents, simple_sents


def filter_sentences(normal,simple,limit):
    norm = []
    simp = []
    for i in range(len(normal)):
        if len(normal[i]) <= limit:
            norm.append(normal[i])
            simp.append(simple[i])
    return norm,simp


def split_data(normal_sents, simple_sents):
    '''

    :param normal_sents:
    :param simple_sents:
    :return: training set and test set
    '''

    shuffle_indices = np.random.permutation(np.arange(len(normal_sents)))
    normal_sents = np.array(normal_sents)[shuffle_indices]
    simple_sents = np.array(simple_sents)[shuffle_indices]
    train_len = int(len(normal_sents) * 0.9)
    normal_sents_train = normal_sents[:train_len]
    simple_sents_train = simple_sents[:train_len]
    normal_sents_test = normal_sents[train_len:]
    simple_sents_test = simple_sents[train_len:]

    return normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test


def build_embedding_matrix(glove_path, vocab):
    '''
    build embedding matrix of size (Vocab_size, embedding_dim) (GloVe)
    returns the embedding matrix and a word-to-index dictionary
    '''

    embeddings_index = dict()
    f = open(glove_path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    word_index = {**vocab.word2id, **vocab.word2id}
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, len(embeddings_index[word])),dtype=np.float32)
    unk_list = []
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            unk_list.append(i)
    # give unk words an average embedding
    avg_embedding = np.average(embedding_matrix, axis=0)
    for i in unk_list:
        embedding_matrix[i] = avg_embedding

    return embedding_matrix


def max_input_sentece_length(normal_sents):
    max_len = max(normal_sents, key=len)
    return len(max_len)


def max_output_sentece_length(simple_sents):
    max_len = max(simple_sents, key=len)
    return len(max_len)


def load_dataset(normal_sents, simple_sents, dataset_size):
    # load training set and test set
    # Cut the datasets randomly
    rand_idx = np.random.choice(len(normal_sents), size=dataset_size)
    normal_sents = np.array(normal_sents)[rand_idx]
    simple_sents = np.array(simple_sents)[rand_idx]
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = split_data(normal_sents,
                                                                                              simple_sents)
    return normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test


def check_and_convert_to_cuda(var,use_cuda):
    return var.cuda() if use_cuda else var


def construct_vocab(sentences, Vocabulary):
    word2id = dict()
    id2word = dict()
    word2id[SOS_TOKEN] = 0
    word2id[EOS_TOKEN] = 1
    id2word[0] = SOS_TOKEN
    id2word[1] = EOS_TOKEN
    for sentence in sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(word2id)-1] = word
    return Vocabulary(word2id, id2word)


# no use for eos at the moment
def sent_to_word_id(sentences, vocab, max_len, eos=True):
    data = []
    for sent in sentences:
        if eos:
            end = [vocab.word2id[EOS_TOKEN]]
        else:
            end = []

        # if len(sent) <= max_len:
        data.append([vocab.word2id[w] for w in sent])

    return data

def word_id_to_sent(ids,vocab):
    words = []
    for id in ids:
        words.append(vocab.id2word[id])
    sent = ' '.join(words)
    return sent
