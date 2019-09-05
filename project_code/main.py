# choose GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import sys
from process_data import *
from model import *
from collections import namedtuple
import torch
import random

VALIDATION_SPLIT = 0.33
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
DROP_PROB = 0.2
MAX_LEN_OF_SENTENCE = 5
FILTER_SIZES = (2, 3, 4)
BATCH_SIZE = 256
NUM_EPOCHES = 1000
CONV_LAYERS = 5
LIMIT_DATA_SIZE = -1
LEARNING_RATE = 0.0001
DATASET_PATH = '../data/'
ACTIVE_DATASET = 'newsela'

# set to false for quicker run time (good for debugging)
LOAD_EMBEDDINGS = True

EVAL_PRINT = 200
PRINT_EVERY = 1


def main():
    assert(len(sys.argv) == 2), 'No GloVe path provided'
    random.seed(0)
    use_cuda = torch.cuda.is_available()

    # data preperation
    normal_sents_orig, simple_sents_orig = load_data(DATASET_PATH, ACTIVE_DATASET, MAX_LEN_OF_SENTENCE,limit_data=LIMIT_DATA_SIZE)

    vocabulary_normal = namedtuple('Vocabulary', ['word2id', 'id2word'])
    vocab_normal = construct_vocab(normal_sents_orig, vocabulary_normal)
    vocabulary_simple = namedtuple('Vocabulary', ['word2id', 'id2word'])
    vocab_simple = construct_vocab(simple_sents_orig, vocabulary_simple)

    word_freq = get_word_frequency(normal_sents_orig + simple_sents_orig)

    normal_sents,simple_sents = filter_sentences(normal_sents_orig,simple_sents_orig,MAX_LEN_OF_SENTENCE)

    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = load_dataset(normal_sents, simple_sents, len(normal_sents_orig))
    simple_max_len = max_output_sentece_length(simple_sents_orig)

    normal_data = sent_to_word_id(normal_sents_train, vocab_normal, simple_max_len)
    simple_data = sent_to_word_id(simple_sents_train, vocab_simple, simple_max_len)

    input_dataset = [Variable(torch.LongTensor(sent)) for sent in normal_data]
    output_dataset = [Variable(torch.LongTensor(sent)) for sent in simple_data]
    if use_cuda: # And if cuda is available use the cuda tensor types
        input_dataset = [i.cuda() for i in input_dataset]
        output_dataset = [i.cuda() for i in output_dataset]

    assert(len(normal_data) == len(simple_data)), 'data length doesnt match'

    # tokenizer = create_tokenizer(normal_sents_orig + simple_sents_orig)
    voc_size_normal = len(vocab_normal.word2id) + 1
    voc_size_simple = len(vocab_simple.word2id) + 1
    glove_path = sys.argv[1]
    embedding_matrix_normal = None
    embedding_matrix_simple = None
    hidden_size = EMBEDDING_DIM
    if LOAD_EMBEDDINGS:
        embedding_matrix_normal = build_embedding_matrix(glove_path, vocab_normal)
        embedding_matrix_simple = build_embedding_matrix(glove_path, vocab_simple)
        hidden_size = embedding_matrix_normal.shape[1]

    # fit network
    print('Creating a model')
    model = Rephraser(EMBEDDING_DIM,simple_max_len, DROP_PROB, hidden_size,
                      BATCH_SIZE, NUM_EPOCHES, vocab_normal, vocab_simple,
                      voc_size_normal, voc_size_simple, word_freq, CONV_LAYERS, LEARNING_RATE, use_cuda,
                      embedding_matrix=(embedding_matrix_normal,embedding_matrix_simple))
    #plot_model(model, to_file='model.png', show_shapes=True)
    print('Fitting the model')
    model.trainIters(input_dataset,output_dataset,print_every=PRINT_EVERY)
    print('Done fitting')
    # # test on some training sequences

    # eval_set_norm = normal_sents_test[:EVAL_PRINT]
    # eval_set_simp = simple_sents_test[:EVAL_PRINT]

    eval_set_norm = normal_sents_train[:EVAL_PRINT]
    eval_set_simp = simple_sents_train[:EVAL_PRINT]

    # for instance - eval_set_norm[i] is [303, 1485, 158, 24, 47, 1486, 1487, 1]
    eval_set_norm = sent_to_word_id(eval_set_norm, vocab_normal, MAX_LEN_OF_SENTENCE)
    eval_set_simp = sent_to_word_id(eval_set_simp, vocab_simple, MAX_LEN_OF_SENTENCE)

    for i in range(min(len(eval_set_norm),len(eval_set_simp))):
        model.evaluate((eval_set_norm[i], eval_set_simp[i]))


if __name__ == '__main__':
    main()
