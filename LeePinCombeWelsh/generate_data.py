from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import LeePinCombeWelsh.configuration as Configuration
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

from LeePinCombeWelsh.utils import readCSV, readFile, embedding

def generate_data():
    print("generate data new")
    document_sentences, document_index = readFile(file=Configuration.get_Document())
    data_similarity = readCSV(file=Configuration.get_File_Similarity(), document_no=len(document_index))


    print("Indexing word vectors..........")
    tokenizer = Tokenizer(num_words=Configuration.get_MAX_NB_WORDS())
    tokenizer.fit_on_texts(document_sentences)
    sequences = tokenizer.texts_to_sequences(document_sentences)


    word_index = tokenizer.word_index
    nb_words = min(len(word_index), Configuration.get_MAX_NB_WORDS())
    Configuration.set_MAX_NB_WORDS(nb_words)

    print("Found %s unique tokens." % len(word_index))

    data = pad_sequences(sequences, maxlen=Configuration.get_MAX_SEQUENCE_LENGTH())

    embeddings_index = embedding(file=Configuration.get_GLOVEFILE())
    embedding_matrix = np.zeros(shape=(Configuration.get_MAX_NB_WORDS() + 1, Configuration.get_EMBEDDING_DIM()))

    for word, i in word_index.items():
        if i > Configuration.get_MAX_NB_WORDS():
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    data_x_a, data_x_b, data_y = sample(data, data_similarity)
    return data_x_a, data_x_b, data_y, embedding_matrix

def sample(data, data_similarity):
    print("sample")
    data_x_a = []
    data_x_b = []
    data_y = []
    # data_similarity = (data_similarity - np.mean(data_similarity))
    for i in range(data_similarity.shape[0]):
        for j in range(data_similarity.shape[1]):
            if data_similarity[i][j] != 0:
                data_x_a.append(data[i])
                data_x_b.append(data[j])
                data_y.append(data_similarity[i][j])
    data_x_a = np.array(data_x_a)
    data_x_b = np.array(data_x_b)
    data_y = np.array(data_y)

    return data_x_a, data_x_b, data_y

