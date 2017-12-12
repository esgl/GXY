from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import csv
import LeePinCombeWelsh.configuration as Configuration

def embedding(file):
    print('embedding')
    embeddings_index = {}
    f = open(file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def readCSV(file, document_no):
    print('readCSV')
    similarity_matrix = np.zeros(shape=[document_no, document_no], dtype='float32')
    # print(similarity_matrix[0:1])
    max_similarity = 0
    with open(file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            document1 = int(row['Document1'])
            document2 = int(row['Document2'])
            similarity = float(row['Similarity'])
            if max_similarity < similarity:
                max_similarity = similarity
            similarity_matrix[document1 - 1][document2 - 1] = similarity

    return similarity_matrix / max_similarity



def readFile(file):
    print('readFile')
    document_sentences = []  #each sentence in document
    document_index = 1 # document index in documents
    document_sentences_index = 1 # sentence index in documents
    documents_index = {} # all sentence indexs in each document
    max_words_in_sentence_length = -1

    with open(file) as f:
        for line in f:
            doc = line.replace('\t','').replace('\n','').split('.',1)[1].split('(')[0].strip()
            sentences = doc.split('.')
            sentences = [sentence for sentence in sentences if len(sentence.strip()) > 1]
            document_sentences_index_tmp = []
            for sentence in sentences:
                document_sentences.append(sentence)
                document_sentences_index_tmp.append(document_sentences_index)
                document_sentences_index += 1
                words_in_sentence_length = len(sentence.split(' '))
                if max_words_in_sentence_length < words_in_sentence_length:
                    max_words_in_sentence_length = words_in_sentence_length
            documents_index[document_index] = document_sentences_index_tmp
            document_index += 1
            Configuration.set_MAX_SEQUENCE_LENGTH(max_words_in_sentence_length)
    return document_sentences, documents_index