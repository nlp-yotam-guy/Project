
from process_data import *
from keras.layers.core import Dropout, Dense
from keras.layers import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils.vis_utils import plot_model
import numpy as np


EMBEDDING_DIM = 100
MAX_LEN_OF_SENTENCE = 192
filter_sizes = (2, 3, 4)
BATCH_SIZE = 32
WIKI_SIMPLE = '/normal.aligned'
WIKI_NORMAL = '/simple.aligned'
GLOVE_PATH = 'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\Glove\\glove.6B.100d.txt'
# wiki_data_path = 'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\wiki'
wiki_data_path = '../data/wiki'

def load_dataset(normal_sents, simple_sents,dataset_size):

    # load training set and test set
    #Cut the datasets randomly
    rand_idx = np.random.choice(len(normal_sents),size=dataset_size)
    normal_sents = np.array(normal_sents)[rand_idx]
    simple_sents = np.array(simple_sents)[rand_idx]
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = split_data(normal_sents, simple_sents)
    return normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test


def Vectorize_Sentences(data, max_len,unk_vec):
    list_of_matrices = np.zeros((len(data),max_len,EMBEDDING_DIM))
    for i, sent in enumerate(data):
        mat = create_sentence_matrix(embedding_matrix, embeddings_matrix_index, sent, unk_vec, max_len)
        list_of_matrices[i] = mat
    return list_of_matrices


def define_model(normal_sent_dataset_size, simple_sent_dataset_size, normal_max_len, simple_max_len, n_units):
    model = Sequential()
    # Need to add here an embadding layer, if we decide to use the tokenizer. In addition if we
    # decide so we need to change this function a little bit.
    model.add(Embedding(normal_sent_dataset_size, n_units, input_length=normal_max_len))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='valid'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(Dense(simple_sent_dataset_size,activation='relu'))
    model.add(LSTM(64))
    model.add(RepeatVector(simple_max_len))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(simple_sent_dataset_size, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, normal_sents_orig, simple_sents_orig):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = simple_sents_orig[i], normal_sents_orig[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        #actual.append([raw_target.split()])
        #predicted.append(translation.split())
    # calculate BLEU score
    #print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == '__main__':
    normal_sents_orig, simple_sents_orig = load_data(wiki_data_path, None)
    #embedding_matrix, embeddings_matrix_index = build_embedding_matrix(GLOVE_PATH)
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = load_dataset(normal_sents_orig, simple_sents_orig,10)
    #unk_vec = create_vector_unknown_token()
    max_len_normal = max_input_sentece_length(normal_sents_orig)
    max_len_simple = max_output_sentece_length(simple_sents_orig)
    #normal_sents_train_as_matrices = Vectorize_Sentences(normal_sents_train,max_len_normal,unk_vec)
    #simple_sents_train_as_matrices = Vectorize_Sentences(simple_sents_train,max_len_simple,unk_vec)
    #normal_sents_test_as_matrices = Vectorize_Sentences(normal_sents_test,max_len_normal,unk_vec)
    #simple_sents_test_as_matrices = Vectorize_Sentences(simple_sents_test,max_len_simple,unk_vec)

    # From here down is a trying of the tokenizer
    normal_tokenizer = create_tokenizer(normal_sents_orig)
    simple_tokenizer = create_tokenizer(simple_sents_orig)
    normal_max_len = max_input_sentece_length(normal_sents_orig)
    simple_max_len = max_output_sentece_length(simple_sents_orig)
    # prepare training data
    train_noraml = encode_sequences(normal_tokenizer, normal_max_len, normal_sents_train)
    train_simple = encode_sequences(simple_tokenizer, simple_max_len, simple_sents_train)
    train_simple = encode_output(train_simple, len(simple_tokenizer.word_index) + 1)
    # prepare validation data
    test_noraml = encode_sequences(normal_tokenizer, normal_max_len, normal_sents_test)
    test_simple = encode_sequences(simple_tokenizer, simple_max_len, simple_sents_test)
    test_simple = encode_output(test_simple, len(simple_tokenizer.word_index) + 1)
    model = define_model(len(normal_tokenizer.word_index) + 1, len(simple_tokenizer.word_index) + 1, normal_max_len, simple_max_len, 64)
    # summarize defined model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    # fit network
    # epochs, batch_size, verbose = 1, 32, 1
    # model.fit(train_noraml, train_simple, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # test on some training sequences
    # print('train')
    # evaluate_model(model, simple_tokenizer, train_noraml, normal_sents_orig, simple_sents_orig)