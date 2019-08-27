# choose GPU
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
from process_data import *
from model import *
from collections import namedtuple
import torch

VALIDATION_SPLIT = 0.33
EMBEDDING_DIM = 100
HIDDEN_SIZE = EMBEDDING_DIM
DROP_PROB = 0.2
MAX_LEN_OF_SENTENCE = 50
FILTER_SIZES = (2, 3, 4)
BATCH_SIZE = 32
NUM_EPOCHES = 1000
CONV_LAYERS = 5
LIMIT_DATA_SIZE = -1
LEARNING_RATE = 0.001
DATASET_PATH = '../data/'
ACTIVE_DATASET = 'newsela'

# set to false for quicker run time (good for debugging)
LOAD_EMBEDDINGS = True

EVAL_PRINT = 20




def main():
    assert(len(sys.argv) == 2), 'No GloVe path provided'

    use_cuda = torch.cuda.is_available()

    # data preperation
    normal_sents_orig, simple_sents_orig = load_data(DATASET_PATH, ACTIVE_DATASET, MAX_LEN_OF_SENTENCE,limit_data=LIMIT_DATA_SIZE)
    normal_sents_train, simple_sents_train, normal_sents_test, simple_sents_test = load_dataset(normal_sents_orig, simple_sents_orig, len(normal_sents_orig))
    simple_max_len = max_output_sentece_length(simple_sents_orig)

    vocabulary = namedtuple('Vocabulary', ['word2id', 'id2word'])
    vocab = construct_vocab(normal_sents_orig + simple_sents_orig, vocabulary)

    normal_data = sent_to_word_id(normal_sents_train, vocab, simple_max_len)
    simple_data = sent_to_word_id(simple_sents_train, vocab, simple_max_len)

    input_dataset = [Variable(torch.LongTensor(sent)) for sent in normal_data]
    output_dataset = [Variable(torch.LongTensor(sent)) for sent in simple_data]
    if use_cuda: # And if cuda is available use the cuda tensor types
        input_dataset = [i.cuda() for i in input_dataset]
        output_dataset = [i.cuda() for i in output_dataset]

    assert(len(normal_data) == len(simple_data)), 'data length doesnt match'

    # tokenizer = create_tokenizer(normal_sents_orig + simple_sents_orig)
    voc_size = len(vocab.word2id) + 1
    glove_path = sys.argv[1]
    embedding_matrix = None
    hidden_size = EMBEDDING_DIM
    if LOAD_EMBEDDINGS:
        embedding_matrix = build_embedding_matrix(glove_path, vocab)
        hidden_size = embedding_matrix.shape[1]



    # train_generator = Batch_Generator(normal_sents_train, simple_sents_train, tokenizer, embedding_matrix, BATCH_SIZE,
    #                                   MAX_LEN_OF_SENTENCE, MAX_LEN_OF_SENTENCE)
    # fit network
    print('Creating a model')
    model = Rephraser(EMBEDDING_DIM,simple_max_len, DROP_PROB, hidden_size,
                      BATCH_SIZE, NUM_EPOCHES, vocab,
                      voc_size, CONV_LAYERS, LEARNING_RATE, use_cuda, embedding_matrix=embedding_matrix)
    #plot_model(model, to_file='model.png', show_shapes=True)
    print('Fitting the model')
    model.trainIters(input_dataset,output_dataset)
    # # test on some training sequences
    print('Training the model')

    eval_set_norm = normal_sents_test[:EVAL_PRINT]
    eval_set_simp = simple_sents_test[:EVAL_PRINT]

    eval_set_norm = sent_to_word_id(eval_set_norm, vocab, MAX_LEN_OF_SENTENCE)
    eval_set_simp = sent_to_word_id(eval_set_simp, vocab, MAX_LEN_OF_SENTENCE)

    for (i, row) in enumerate(zip(eval_set_norm,eval_set_simp)):
        model.evaluate((eval_set_norm[i], eval_set_simp[i]), vocab)

if __name__ == '__main__':
    main()