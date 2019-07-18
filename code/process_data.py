import numpy as np


WIKI_SIMPLE = '/normal.aligned'
WIKI_NORMAL = '/simple.aligned'
GLOVE_PATH = '/home/yotam/personal/NLP/Project/code/glove.6B.100d.txt'
BATCH_SIZE = 32

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
    f_wiki_simple = open(wiki_path + WIKI_SIMPLE,'r')
    f_wiki_normal = open(wiki_path + WIKI_NORMAL,'r')

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



def create_sentence_matrix(embedding_matrix,embeddings_matrix_index,sentence,max_sent_len):
    '''
    embedding a sentence to a matrix of size (max_sent_len, embedding_dim)
    '''
    sent_matrix = np.zeros((max_sent_len, embedding_matrix.shape[1]))
    for i, word in enumerate(sentence):
        if word in embeddings_matrix_index:
            sent_matrix[i] = embedding_matrix[embeddings_matrix_index[word]]
        else:
            sent_matrix[i] = embedding_matrix[embeddings_matrix_index['<unk>']]
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



if __name__=='__main__':
    normal_sents,simple_sents = load_data('/home/yotam/personal/NLP/Project/data/wiki',None)
    embedding_matrix, embeddings_matrix_index = build_embedding_matrix(GLOVE_PATH)
    max_len = max_input_sentece_length(normal_sents)
    sent = ["guy", "is", "great", "in", "soccer"]
    mat = create_sentence_matrix(embedding_matrix,embeddings_matrix_index,sent,max_len)
    print('d')