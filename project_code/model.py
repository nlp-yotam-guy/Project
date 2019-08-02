
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

PRINT_PROGRESS = 1
MAX_EVAL_PRINT = 15


class Rephraser:

    def __init__(self,embed_dim, max_input_len, drop_prob,
                 hidden_size, batch_size, n_epoches, max_output_len,
                 in_vocab_size, out_vocab_size):

        self.embed_dim = embed_dim
        self.max_input_len = max_input_len
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.max_output_len = max_output_len
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size

        self.model = None
        self.define()
        print(self.model.summary())

    def define(self):
        self.model = Sequential()
        self.model.add(Embedding(self.in_vocab_size, self.hidden_size, input_length=self.max_input_len))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='valid'))
        self.model.add(Dropout(self.drop_prob))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        # The number in this LSTM can be any number
        # model.add(LSTM(64))
        self.model.add(RepeatVector(self.max_output_len))
        self.model.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(self.out_vocab_size, activation='softmax')))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def train(self,X,Y,validation_split):
        self.model.fit(X, Y, validation_split=validation_split, epochs=self.n_epoches, batch_size=self.batch_size,
                  verbose=PRINT_PROGRESS)


    # simplify a given sentence
    def predict(self,tokenizer,sentence):
        prediction = self.model.predict(sentence)[0]
        integers = [np.argmax(vector) for vector in prediction]
        target = list()
        for i in integers:
            word = idx2word(i, tokenizer)
            if word is None:
                break
            target.append(word)
        return ' '.join(target)

    # evaluate the the model
    def evaluate(self, tokenizer, sources, normal_sents_orig, simple_sents_orig):
        actual, predicted = list(), list()
        for i, source in enumerate(sources):
            # translate encoded source text
            source = source.reshape((1, source.shape[0]))
            translation = self.predict(tokenizer, source)
            raw_target, raw_src = normal_sents_orig[i], simple_sents_orig[i]
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
            if i >= MAX_EVAL_PRINT:
                break
        # actual.append([raw_target.split()])
        # predicted.append(translation.split())
        # calculate BLEU score
        # print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        # print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        # print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        # print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))




def Vectorize_Sentences(data, max_len,unk_vec):
    list_of_matrices = np.zeros((len(data),max_len,EMBEDDING_DIM))
    for i, sent in enumerate(data):
        mat = create_sentence_matrix(embedding_matrix, embeddings_matrix_index, sent, unk_vec, max_len)
        list_of_matrices[i] = mat
    return list_of_matrices

# map an integer to a word
def idx2word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def define_model2(normal_sent_dataset_size, simple_sent_dataset_size, normal_max_len, simple_max_len, n_units):
    deep_inputs = Input(shape=(normal_max_len,))
    input_layer = Embedding(normal_sent_dataset_size, n_units, input_length=normal_max_len)(deep_inputs)
    conv1 = Conv1D(100, (3), activation='relu')(input_layer)
    dropout_1 = Dropout(0.7)(conv1)
    conv2 = Conv1D(100, (4), activation='relu')(input_layer)
    dropout_2 = Dropout(0.7)(conv2)
    conv3 = Conv1D(100, (5), activation='relu')(input_layer)
    dropout_3 = Dropout(0.7)(conv3)
    conv4 = Conv1D(100, (6), activation='relu')(input_layer)
    dropout_4 = Dropout(0.7)(conv4)
    maxpool1 = MaxPooling1D(pool_size=normal_max_len - 2)(dropout_1)
    maxpool2 = MaxPooling1D(pool_size=normal_max_len - 3)(dropout_2)
    maxpool3 = MaxPooling1D(pool_size=normal_max_len - 4)(dropout_3)
    maxpool4 = MaxPooling1D(pool_size=normal_max_len - 5)(dropout_4)
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




