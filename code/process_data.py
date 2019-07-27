import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


WIKI_SIMPLE = '/normal.aligned'
WIKI_NORMAL = '/simple.aligned'
GLOVE_PATH = 'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\Glove\\glove.6B.100d.txt'
BATCH_SIZE = 32
EMBEDDING_DIM = 100
def load_data(wiki_path,newsela_path):
    '''
    load raw text from data files for normal and simple sentences

    example:
    sentences = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
    '''
    f_wiki_simple = open(wiki_path + WIKI_SIMPLE,'r',encoding="utf8")
    f_wiki_normal = open(wiki_path + WIKI_NORMAL,'r',encoding="utf8")

    normal_sents = f_wiki_normal.readlines()
    simple_sents = f_wiki_simple.readlines()

    assert(len(normal_sents) == len(simple_sents))
    n = len(normal_sents)

    for i in range(n):
        normal_line = normal_sents[i].split('\t')
        normal_sents[i] = normal_line[2].split(' ')
        del normal_sents[i][-1]
        normal_sents[i].append('.')

        simple_line = simple_sents[i].split('\t')
        simple_sents[i] = simple_line[2].split(' ')
        del simple_sents[i][-1]
        simple_sents[i].append('.')

    return normal_sents,simple_sents


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

def split_data(normal_sents,simple_sents):
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

def build_embedding_matrix(glove_path):
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
    embedding_matrix = np.zeros((len(embeddings_index), 100))
    i=0
    embeddings_matrix_index = dict()
    for word in embeddings_index:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embeddings_matrix_index[word] = i
        i +=1

    return embedding_matrix, embeddings_matrix_index


def max_input_sentece_length(normal_sents):
    max_len = max(normal_sents,key=len)
    return len(max_len)


def max_output_sentece_length(simple_sents):
    max_len = max(simple_sents,key=len)
    return len(max_len)


def get_vocab_size(embedding_matrix):
    return embedding_matrix.shape[0]


# def pad_sentences(sentences):
#     max_len = max_input_sentece_length(sentences)
#     for sent in sentences:
#         need_to_pad = max_len - len(sent)
#         sent.extend([0]*need_to_pad)
#     return



def create_sentence_matrix(embedding_matrix,embeddings_matrix_index,sentence,unk_vector, max_len):
    '''
    embedding a sentence to a matrix of size (max_sent_len, embedding_dim)
    '''

    #sent_matrix = []
    sent_matrix = np.zeros((max_len, embedding_matrix.shape[1]))
    for i, word in enumerate(sentence):
        if word.lower() in embeddings_matrix_index:
            sent_matrix[i] = np.array(embedding_matrix[embeddings_matrix_index[word.lower()]])
        else:
            sent_matrix[i] = np.array(unk_vector)
    return sent_matrix



def create_batch_matrix(embedding_matrix, embeddings_matrix_index, normal_sents, simple_sents, max_len_simple, max_len_normal):
    '''
    embedding an input batch of sentences to a 3d matrix of size (BATCH_SIZE, max_len_normal, embedding_dim)
    and an ouput 'gold' 3d matrix of size (BATCH_SIZE, max_len_simple, embedding_dim)
    '''

    batch = np.zeros([BATCH_SIZE, max_len_normal,embeddings_matrix_index.shape[1]])
    simplified = np.zeros([BATCH_SIZE, max_len_simple,embeddings_matrix_index.shape[1]])

    # Construct the data batch and run you backpropogation implementation
    ### YOUR CODE HERE

    rand_idx = np.random.choice(len(normal_sents),size=BATCH_SIZE)

    for i in range(BATCH_SIZE):
        batch[i] = create_sentence_matrix(embedding_matrix,embeddings_matrix_index,normal_sents[rand_idx[i]], max_len_normal)
        simplified[i]   = create_sentence_matrix(embedding_matrix,embeddings_matrix_index,simple_sents[rand_idx[i]], max_len_simple)
    
    return batch,simplified


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
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

if __name__=='__main__':
    '''
    normal_sents,simple_sents = load_data('C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\wiki',None)
    embedding_matrix, embeddings_matrix_index = build_embedding_matrix(GLOVE_PATH)
    max_len = max_input_sentece_length(normal_sents)
    sent = ["guy", "is", "great", "in", "soccer"]
    mat = create_sentence_matrix(embedding_matrix,embeddings_matrix_index,sent,max_len)
    
    '''
    normal_sents, simple_sents = load_data('C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\wiki', None)
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = split_data(normal_sents, simple_sents)
    normal_tokenizer = create_tokenizer(normal_sents)
    simple_tokenizer = create_tokenizer(simple_sents)
    normal_max_len = max_input_sentece_length(normal_sents)
    simple_max_len = max_output_sentece_length(simple_sents)

    print('d')