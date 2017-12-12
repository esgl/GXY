from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

import newsgroup20.configuration as Configuration
from keras.preprocessing.text import Tokenizer
# import keras.preprocessing.text
# from keras.preprocessing.text import Tokenizer as tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout, BatchNormalization

from newsgroup20.utils import embedding_index, corpus, corpus_categories
def generate_data():

    data, data_pairs = corpus_categories()
    print(len(data))
    print(len(data_pairs))
    # documents = []
    documents = data
    # for value in data.values():
    #     documents.append(value)
    tokenizer = Tokenizer(num_words=Configuration.get_MAX_NB_WORDS())
    tokenizer.fit_on_texts(documents)
    sequences = tokenizer.texts_to_sequences(documents)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences=sequences, maxlen=Configuration.get_MAX_SEQUENCE_LENGTH())
    # print(data.shape)
    # print(data[1:2,:])


    nb_words = min(len(word_index), Configuration.get_MAX_NB_WORDS())
    Configuration.set_MAX_NB_WORDS(nb_words)

    embeddings_index = embedding_index()
    embedding_matrix = np.zeros(shape=(Configuration.get_MAX_NB_WORDS() + 1, Configuration.get_EMBEDDING_DIM()))
    for word, i in word_index.items():
        if i > Configuration.get_MAX_NB_WORDS():
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    document_a = []
    document_b = []
    category = []
    for key, value in data_pairs.items():
        document_pair = key.split("_")
        document_a_index = int(document_pair[0])
        document_b_index = int(document_pair[1])
        document_a.append(data[document_a_index])
        document_b.append(data[document_b_index])
        category.append(value)
    document_a = np.array(document_a)
    document_b = np.array(document_b)
    category = np.array(category)

    print(len(document_a))
    print(len(document_b))
    print(len(category))
    (document_a_train, document_b_train, category_train,
     document_a_test, document_b_test, category_test) \
        = sample(document_a=document_a, document_b=document_b, category=category)

    return (document_a_train, document_b_train, category_train, document_a_test, document_b_test, category_test), embedding_matrix

def sample(document_a, document_b, category):
    indices = np.arange(document_a.shape[0])
    # print(indices)
    # exit()
    np.random.shuffle(indices)

    document_a = document_a[indices]
    document_b = document_b[indices]
    category = category[indices]

    nb_validation_sample = int(Configuration.get_VALIDATION_SPLIT() * document_a.shape[0])

    document_a_train = document_a[:-nb_validation_sample]
    document_a_test = document_a[-nb_validation_sample:]
    document_b_train = document_b[:-nb_validation_sample]
    document_b_test = document_b[-nb_validation_sample:]
    category_train = category[:-nb_validation_sample]
    category_test = category[-nb_validation_sample:]

    return (document_a_train, document_b_train, category_train,
            document_a_test, document_b_test, category_test)

if __name__ == "__main__":
    generate_data()
    # documents = []
    # labels = []
    # for key, value in data.items():
    #     documents.append(key)
    #     labels.append(value)
    #
    # tokenizer = Tokenizer(num_words=Configuration.get_MAX_NB_WORDS())
    # tokenizer.fit_on_texts(documents)
    # sequences = tokenizer.texts_to_sequences(documents)
    #
    # word_index = tokenizer.word_index
    #
    # data = pad_sequences(sequences, maxlen=Configuration.get_MAX_SEQUENCE_LENGTH())
    # labels = to_categorical(np.asarray(labels))
    # print(labels)
    # exit()
    # print('Shape of data tensor: ', data.shape)
    # print('Shape of label tensor: ', labels.shape)
    #
    # indices = np.asarray(data.shape[0])
    # np.random.shuffle(indices)
    #
    # data = data[indices]
    # labels = labels[indices]
    # nb_validation_sample = int(Configuration.get_VALIDATION_SPLIT() * data.shape[0])
    #
    # x_train = data[:-nb_validation_sample]
    # y_train = data[:-nb_validation_sample]
