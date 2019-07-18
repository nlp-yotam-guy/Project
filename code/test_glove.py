import numpy as np

#Creating an embedding matrix
sent = ["Guy", "is", "great", "in", "soccer"]
# load the whole embedding into memory
embeddings_index = dict()
f = open('C:\\Users\\GAZOV\\Desktop\\glove.6B\\glove.6B.100d.txt', encoding="utf8")
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
# Finish creating an embedding matrix

#Test:
our_sent_matrix = np.zeros((5, 100))
for i, word in enumerate(sent):
    if word in embeddings_matrix_index:
        our_sent_matrix[i] = embedding_matrix[embeddings_matrix_index[word]]
    else:
        our_sent_matrix[i] = embedding_matrix[embeddings_matrix_index['<unk>']]
print("we can go home")