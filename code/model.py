
from process_data import *
from keras.layers.core import Dropout, Dense
from keras.layers import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM, Bidirectional, concatenate, Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import Model, Input
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

def define_model(normal_sent_dataset_size, simple_sent_dataset_size, normal_max_len, simple_max_len, n_units):
    model = Sequential()
    model.add(Embedding(normal_sent_dataset_size, n_units, input_length=normal_max_len))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='valid'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # The number in this LSTM can be any number
    #model.add(LSTM(64))
    model.add(RepeatVector(simple_max_len))
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(simple_sent_dataset_size, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_model2(normal_sent_dataset_size, simple_sent_dataset_size, normal_max_len, simple_max_len, n_units):
    deep_inputs = Input(shape=(normal_max_len,))
    input_layer = Embedding(normal_sent_dataset_size, n_units, input_length=normal_max_len)(deep_inputs)
    conv1 = Conv1D(100, (13), activation='relu')(input_layer)
    dropout_1 = Dropout(0.7)(conv1)
    conv2 = Conv1D(100, (14), activation='relu')(input_layer)
    dropout_2 = Dropout(0.7)(conv2)
    conv3 = Conv1D(100, (15), activation='relu')(input_layer)
    dropout_3 = Dropout(0.7)(conv3)
    conv4 = Conv1D(100, (16), activation='relu')(input_layer)
    dropout_4 = Dropout(0.7)(conv4)
    maxpool1 = MaxPooling1D(pool_size=normal_max_len - 12)(dropout_1)
    maxpool2 = MaxPooling1D(pool_size=normal_max_len - 13)(dropout_2)
    maxpool3 = MaxPooling1D(pool_size=normal_max_len - 14)(dropout_3)
    maxpool4 = MaxPooling1D(pool_size=normal_max_len - 15)(dropout_4)
    flat1 = Flatten()(maxpool1)
    flat2 = Flatten()(maxpool2)
    flat3 = Flatten()(maxpool3)
    flat4 = Flatten()(maxpool4)
    cc1 = concatenate([flat1,flat2,flat3,flat4])
    vec = RepeatVector(simple_max_len)(cc1)
    lstm = LSTM(236, return_sequences=True)(vec)
    output = Dense(simple_sent_dataset_size, activation='softmax')(lstm)
    model = Model(inputs=[deep_inputs], outputs=output)
    learning_rate = 1e-3
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    #print(model.summary())
    return model


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, normal_sents_orig, simple_sents_orig):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = normal_sents_orig[i], simple_sents_orig[i]
        if i < 15:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
    #actual.append([raw_target.split()])
        #predicted.append(translation.split())
    # calculate BLEU score
    #print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == '__main__':

    # data preperation
    normal_sents_orig, simple_sents_orig = load_data(wiki_data_path, None)
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = load_dataset(normal_sents_orig, simple_sents_orig, 10000)
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
    # fit network
    print('Creating a model')
    model = define_model2(len(normal_tokenizer.word_index) + 1, len(simple_tokenizer.word_index) + 1, normal_max_len, simple_max_len, 64)
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    epochs, batch_size, verbose = 30, 32, 1
    print('Fitting the model')
    model.fit(train_noraml, train_simple, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # test on some training sequences
    print('Training the model')
    evaluate_model(model, simple_tokenizer, train_noraml, normal_sents_orig, simple_sents_orig)
