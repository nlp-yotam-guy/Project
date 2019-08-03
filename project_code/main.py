# choose GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import sys
from process_data import *
from model import Rephraser

VALIDATION_SPLIT = 0.33
EMBEDDING_DIM = 100
HIDDEN_SIZE = EMBEDDING_DIM
DROP_PROB = 0.2
MAX_LEN_OF_SENTENCE = 192
FILTER_SIZES = (2, 3, 4)
BATCH_SIZE = 32
NUM_EPOCHES = 10
LIMIT_DATA_SIZE = 10000
WIKI_SIMPLE = '../data/wiki/normal.aligned'
WIKI_NORMAL = '../data/wiki/simple.aligned'
GLOVE_PATH = 'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\Glove\\glove.6B.100d.txt'
# wiki_data_path = 'C:\\Users\\guyazov\\PycharmProjects\\SentenceSimplificationProject\\data\\wiki'



def main():
    assert(len(sys.argv) == 2, 'No GloVe path provided')

    # data preperation
    normal_sents_orig, simple_sents_orig = load_data(WIKI_NORMAL,WIKI_SIMPLE, None)
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = load_dataset(normal_sents_orig, simple_sents_orig, len(normal_sents_orig))

    tokenizer = create_tokenizer(normal_sents_orig + simple_sents_orig)
    glove_path = sys.argv[1]
    embedding_matrix = build_embedding_matrix(glove_path, tokenizer)

    # simple_tokenizer = create_tokenizer(simple_sents_orig)
    normal_max_len = max_input_sentece_length(normal_sents_orig)
    simple_max_len = max_output_sentece_length(simple_sents_orig)
    # prepare training data
    train_noraml = encode_sequences(tokenizer, normal_max_len, normal_sents_train)
    train_simple = encode_sequences(tokenizer, simple_max_len, simple_sents_train)
    train_simple = encode_output(train_simple, len(tokenizer.word_index) + 1)
    # prepare validation data
    test_normal = encode_sequences(tokenizer, normal_max_len, normal_sents_test)
    test_simple = encode_sequences(tokenizer, simple_max_len, simple_sents_test)
    test_simple = encode_output(test_simple, len(tokenizer.word_index) + 1)
    # fit network
    print('Creating a model')
    model = Rephraser(EMBEDDING_DIM, embedding_matrix, normal_max_len, DROP_PROB, HIDDEN_SIZE,
                      BATCH_SIZE, NUM_EPOCHES, simple_max_len,
                      embedding_matrix.shape[0])

    #plot_model(model, to_file='model.png', show_shapes=True)
    print('Fitting the model')
    model.train(train_noraml, train_simple, VALIDATION_SPLIT)
    # # test on some training sequences
    print('Training the model')
    model.evaluate(tokenizer, train_noraml, normal_sents_orig, simple_sents_orig)


if __name__ == '__main__':
    main()