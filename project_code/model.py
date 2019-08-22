from process_data import *
from attention import *
from keras.layers import *
from keras.layers.core import Dropout, Dense
from keras.layers import Conv1D, Activation
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM, Bidirectional, concatenate, Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import Model, Input
import tensorflow as tf
import numpy as np

from MLSTM import MLSTM

PRINT_PROGRESS = 1
MAX_EVAL_PRINT = 15


class Rephraser:

    def __init__(self,embed_dim, max_input_len, drop_prob,
                 hidden_size, batch_size, n_epoches, max_output_len,
                 vocab_size,n_conv_layers,kernel_size=3,embedding_matrix=None):

        self.embed_dim = embed_dim
        self.embedding_matrix = embedding_matrix
        self.max_input_len = max_input_len
        self.normal_max_len = max_input_len
        self.simple_max_len = max_output_len
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.max_output_len = max_output_len
        self.vocab_size = vocab_size
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size

        if embedding_matrix is None:
            self.embedding_matrix = np.zeros((self.vocab_size,self.embed_dim))

        self.model = None
        # self.define()
        # self.define_nmt()
        #self.define_model2()
        self.CNN_LSTM()
        print(self.model.summary())

    def define(self):
        # https://github.com/pradeepsinngh/Neural-Machine-Translation/blob/master/machine_translation.ipynb
        learning_rate = 1e-3

        input_seq = Input(batch_shape=(self.batch_size,self.max_input_len),dtype='int32')
        emb = Embedding(self.vocab_size,
                        self.embed_dim,
                        weights=[self.embedding_matrix],
                        trainable=False)(input_seq)
        bdrnn = LSTM(self.hidden_size, return_sequences=True)(emb)
        dense = Dense(self.vocab_size, activation='softmax')
        logits = TimeDistributed(dense)(bdrnn)

        self.model = Model(inputs=input_seq, outputs=logits)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])


    def define_nmt(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.normal_max_len,))
        decoder_inputs = Input(shape=(self.simple_max_len,))
        # Embedding layer
        encoder_embeddings = Embedding(self.vocab_size,
                                       self.hidden_size,
                                       input_length=self.normal_max_len,
                                       weights=[self.embedding_matrix],
                                       trainable=True)(encoder_inputs)

        decoder_embeddings = Embedding(self.vocab_size,
                                       self.hidden_size,
                                       input_length=self.simple_max_len,
                                       weights=[self.embedding_matrix],
                                       trainable=True)(decoder_inputs)
        # Convolutional Encoder
        encoder_conv = Conv1D(filters=self.hidden_size, kernel_size=3, activation='relu', padding='valid')
        ### problem here
        encoder_out = encoder_conv(encoder_embeddings)
        # Set up the decoder GRU, using `encoder_states` as initial state.
        decoder_lstm = LSTM(self.hidden_size, return_sequences=True)
        decoder_out = decoder_lstm(decoder_embeddings)
        # Attention layer
        attention = dot([decoder_out, encoder_out], axes=[2, 2])
        attention = Activation('softmax')(attention)
        # Concat attention output and decoder output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attention])
        # Dense layer
        dense = Dense(self.vocab_size, activation='softmax', name='softmax_layer')
        dense_time = TimeDistributed(dense, name='time_distributed_layer')
        decoder_pred = dense_time(decoder_concat_input)
        # Full model
        full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
        full_model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model = full_model


    def define_model2(self):
        deep_inputs = Input(shape=(self.normal_max_len,))
        input_layer = Embedding(self.vocab_size, self.hidden_size, input_length=self.normal_max_len)(deep_inputs)
        conv1 = Conv1D(100, (3), activation='relu')(input_layer)
        dropout_1 = Dropout(0.7)(conv1)
        conv2 = Conv1D(100, (4), activation='relu')(input_layer)
        dropout_2 = Dropout(0.7)(conv2)
        conv3 = Conv1D(100, (5), activation='relu')(input_layer)
        dropout_3 = Dropout(0.7)(conv3)
        conv4 = Conv1D(100, (6), activation='relu')(input_layer)
        dropout_4 = Dropout(0.7)(conv4)
        maxpool1 = MaxPooling1D(pool_size=self.normal_max_len - 2)(dropout_1)
        maxpool2 = MaxPooling1D(pool_size=self.normal_max_len - 3)(dropout_2)
        maxpool3 = MaxPooling1D(pool_size=self.normal_max_len - 4)(dropout_3)
        maxpool4 = MaxPooling1D(pool_size=self.normal_max_len - 5)(dropout_4)
        flat1 = Flatten()(maxpool1)
        flat2 = Flatten()(maxpool2)
        flat3 = Flatten()(maxpool3)
        flat4 = Flatten()(maxpool4)
        cc1 = concatenate([flat1, flat2, flat3, flat4])
        vec = RepeatVector(self.simple_max_len)(cc1)
        lstm = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(vec)
        attention = dot([lstm, vec], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, vec], axes=[2, 1])
        output = Dense(self.vocab_size, activation='softmax')(context)
        model = Model(inputs=[deep_inputs], outputs=output)
        learning_rate = 1e-3
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
        self.model = model


    def CNN_LSTM(self):

        # encoder
        inputs_sentence = Input(batch_shape=(self.batch_size,self.max_input_len,))
        input_positions = Input(batch_shape=(self.batch_size,self.max_input_len,))
        embed_sentence = Embedding(self.vocab_size,
                                   self.hidden_size,
                                   input_length=self.normal_max_len,
                                   weights=[self.embedding_matrix],
                                   trainable=False)(inputs_sentence)

        embed_position = Embedding(self.vocab_size,
                                   self.hidden_size,
                                   input_length=self.normal_max_len,
                                   trainable=True)(input_positions)

        embedding = Dropout(self.drop_prob)(embed_sentence + embed_position)

        conv_a = Conv1D(self.hidden_size,self.kernel_size,padding='same')(embedding)
        for i in range(1,self.n_conv_layers):
            conv_a = conv_a + Conv1D(self.hidden_size,self.kernel_size,padding='same')(conv_a)
            conv_a = Activation('tanh')(conv_a)

        conv_c = Conv1D(self.hidden_size, self.kernel_size, padding='same')(embedding)
        for i in range(1, self.n_conv_layers):
            conv_c = conv_c + Conv1D(self.hidden_size, self.kernel_size, padding='same')(conv_c)
            conv_c = Activation('tanh')(conv_c)

        ##### end of encoding #####

        z = conv_a
        z_tag = conv_c

        # decoder
        lstm = MLSTM(self.hidden_size, return_state=True, return_sequences=True)
        decoded = lstm.call(z,z_tag)
        dense = Dense(self.vocab_size, activation='softmax')
        logits = TimeDistributed(dense)(decoded)

        self.model = Model(inputs=[inputs_sentence,input_positions], outputs=logits)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])

        # states = None
        # for i in range(z.shape[1]):
        #     lstm_single_timestep = z[:,i,:]
        #     shape = (int(lstm_single_timestep.shape[0]),1,int(lstm_single_timestep.shape[1]))
        #     lstm_single_timestep = tf.reshape(lstm_single_timestep, shape)
        #     if states is None:
        #         lstm_single_timestep, state_h, state_c = \
        #             LSTM(self.hidden_size, return_state=True, return_sequences=True)(lstm_single_timestep)
        #     else:
        #         lstm_single_timestep, state_h, state_c = \
        #             LSTM(self.hidden_size, return_state=True,
        #                  return_sequences=True)(lstm_single_timestep,initial_state=states)
        #     states = [state_h, state_c]
        #     attention = Activation('softmax')(dot([state_h,lstm_single_timestep],-1))
        #     c_i = dot([attention,z],-2)



        print('a')




    def train(self,generator,validation_split):
        self.model.fit_generator(generator, epochs=self.n_epoches, verbose=PRINT_PROGRESS)


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





