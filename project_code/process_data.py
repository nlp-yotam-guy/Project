import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence

# WIKI_SIMPLE = '/normal.aligned'
# WIKI_NORMAL = '/simple.aligned'
# GLOVE_PATH = 'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\Glove\\glove.6B.100d.txt'
# BATCH_SIZE = 32
# EMBEDDING_DIM = 100

WIKI_SIMPLE = '/normal.aligned'
WIKI_NORMAL = '/simple.aligned'
NEWSELA = '/newsela_articles_20150302.aligned.sents.txt'


class Batch_Generator(Sequence):
    def __init__(self, normal, simple, tokenizer, embedding_matrix, batch_size, max_len_normal, max_len_simple):
        self.normal = normal
        self.simple = simple
        self.tokenizer = tokenizer
        self.embedding_matrix = embedding_matrix
        self.batch_size = batch_size
        self.max_len_normal = max_len_normal
        self.max_len_simple = max_len_simple

    def __len__(self):
        return int(np.ceil(len(self.normal) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.normal[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.simple[idx * self.batch_size: (idx + 1) * self.batch_size]

        train_normal = encode_sequences(self.tokenizer, self.max_len_normal, batch_x)
        train_simple = encode_sequences(self.tokenizer, self.max_len_simple, batch_y)
        train_simple = encode_output(train_simple, len(self.tokenizer.word_index) + 1)

        # return (train_normal, np.arange(len(train_normal))), train_simple
        return train_normal, train_simple


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
        normal_sent = normal_line.split('\t')[-1].lower()
        simple_sent = simple_line.split('\t')[-1].lower()

        if normal_sent == simple_sents:
            continue

        normal_sent = normal_sent.split(' ')
        if len(normal_sent) > limit_sent_len:
            continue

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

    newsela = '/home/yotam/personal/NLP/Project/data/newsela/newsela_articles_20150302.aligned.sents.txt'
    f = open(newsela, 'r')
    for line in f:
        if i > limit_data:
            break
        splited_line = line.split('\t')
        if splited_line[-2] == splited_line[-1]:
            continue
        normal_sent = splited_line[-2].lower().split(' ')
        if len(normal_sent) > limit_sent_len:
            continue
        if normal_sent[-1] != '.':
            normal_sent.append('.')

        simple_sent = splited_line[-1].lower().split(' ')
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


def create_vector_unknown_token():
    '''
    :return: a vector for unknown token
    '''
    # Get number of vectors and hidden dim
    with open(GLOVE_PATH, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            pass
    n_vec = i + 1
    hidden_dim = len(line.split(' ')) - 1

    vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

    with open(GLOVE_PATH, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

    average_vec = np.mean(vecs, axis=0)
    return average_vec


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


def build_embedding_matrix(glove_path, tokenizer):
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
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, len(embeddings_index[word])))
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


def get_vocab_size(embedding_matrix):
    return embedding_matrix.shape[0]


# def pad_sentences(sentences):
#     max_len = max_input_sentece_length(sentences)
#     for sent in sentences:
#         need_to_pad = max_len - len(sent)
#         sent.extend([0]*need_to_pad)
#     return


def create_sentence_matrix(embedding_matrix, embeddings_matrix_index, sentence, unk_vector, max_len):
    '''
    embedding a sentence to a matrix of size (max_sent_len, embedding_dim)
    '''

    # sent_matrix = []
    sent_matrix = np.zeros((max_len, embedding_matrix.shape[1]))
    for i, word in enumerate(sentence):
        if word.lower() in embeddings_matrix_index:
            sent_matrix[i] = np.array(embedding_matrix[embeddings_matrix_index[word.lower()]])
        else:
            sent_matrix[i] = np.array(unk_vector)
    return sent_matrix


def create_batch_matrix(embedding_matrix, embeddings_matrix_index, normal_sents, simple_sents, max_len_simple,
                        max_len_normal):
    '''
    embedding an input batch of sentences to a 3d matrix of size (BATCH_SIZE, max_len_normal, embedding_dim)
    and an ouput 'gold' 3d matrix of size (BATCH_SIZE, max_len_simple, embedding_dim)
    '''

    batch = np.zeros([BATCH_SIZE, max_len_normal, embeddings_matrix_index.shape[1]])
    simplified = np.zeros([BATCH_SIZE, max_len_simple, embeddings_matrix_index.shape[1]])

    # Construct the data batch and run you backpropogation implementation
    ### YOUR CODE HERE

    rand_idx = np.random.choice(len(normal_sents), size=BATCH_SIZE)

    for i in range(BATCH_SIZE):
        batch[i] = create_sentence_matrix(embedding_matrix, embeddings_matrix_index, normal_sents[rand_idx[i]],
                                          max_len_normal)
        simplified[i] = create_sentence_matrix(embedding_matrix, embeddings_matrix_index, simple_sents[rand_idx[i]],
                                               max_len_simple)

    return batch, simplified


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape((sequences.shape[0], sequences.shape[1], vocab_size))
    return y


def load_dataset(normal_sents, simple_sents, dataset_size):
    # load training set and test set
    # Cut the datasets randomly
    rand_idx = np.random.choice(len(normal_sents), size=dataset_size)
    normal_sents = np.array(normal_sents)[rand_idx]
    simple_sents = np.array(simple_sents)[rand_idx]
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = split_data(normal_sents,
                                                                                              simple_sents)
    return normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test


if __name__ == '__main__':
    '''
    normal_sents,simple_sents = load_data('C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\wiki',None)
    embedding_matrix, embeddings_matrix_index = build_embedding_matrix(GLOVE_PATH)
    max_len = max_input_sentece_length(normal_sents)
    sent = ["guy", "is", "great", "in", "soccer"]
    mat = create_sentence_matrix(embedding_matrix,embeddings_matrix_index,sent,max_len)

    '''
    normal_sents, simple_sents = load_data(
        'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\wiki', None)
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = split_data(normal_sents,
                                                                                              simple_sents)
    normal_tokenizer = create_tokenizer(normal_sents)
    simple_tokenizer = create_tokenizer(simple_sents)
    normal_max_len = max_input_sentece_length(normal_sents)
    simple_max_len = max_output_sentece_length(simple_sents)

    print('d')