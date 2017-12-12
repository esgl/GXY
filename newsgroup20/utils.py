from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import pickle

import newsgroup20.configuration as Configuration

from sklearn.datasets import fetch_20newsgroups
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as S_Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summerizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def embedding_index():
    print("embedding_index..........................................")
    embeddings_weights = {}

    if not os.path.exists(Configuration.get_EMBEDDING_INDEX_FILE()):
        f = open(Configuration.get_GLOVE_FILE())
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_weights[word] = coefs
        f.close()
        pickle_embeddings_index = open(Configuration.get_EMBEDDING_INDEX_FILE(), "wb")
        pickle.dump(embeddings_weights, pickle_embeddings_index)
        pickle_embeddings_index.close()
    else:
        pickle_embeddings_index = open(Configuration.get_EMBEDDING_INDEX_FILE(), "rb")
        embeddings_weights = pickle.load(pickle_embeddings_index)
        pickle_embeddings_index.close()
    return embeddings_weights


def corpus_categories():
    print("corpus_categories.........................................")
    categories = ['alt.atheism', 'comp.graphics']
    newsgroups_data = fetch_20newsgroups(subset="all", categories=categories)
    # print(len(newsgroups_data.data))
    # exit()
    md5_str = md5("".join(categories))
    if not os.path.exists(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR()):
        os.mkdir(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR())
    data = []
    data_pair = {}
    if Configuration.get_SUMMERIZED():
        pkl_file = os.path.join(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR(),
                                str(Configuration.get_SENTENCES_COUNT()) + "_" + str(md5_str) +
                                "_" + str(Configuration.get_SUBDATA_LENGTH()) + "_sumy.pkl")
        pkl_file_pair = os.path.join(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR(),
                                str(Configuration.get_SENTENCES_COUNT()) + "_" + str(md5_str) +
                                     "_" + str(Configuration.get_SUBDATA_LENGTH()) + "_sumy_pair.pkl")
    else:
        pkl_file = os.path.join(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR(), "original.pkl")
        pkl_file_pair = os.path.join(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR(), "original_pair.pkl")

    if not os.path.exists(pkl_file):
        newsgroups_data_document = newsgroups_data.data
        newsgroups_data_target = newsgroups_data.target
        newsgroups_data_len = len(newsgroups_data_document)



        if Configuration.get_SUBDATA_LENGTH() > newsgroups_data_len:
            Configuration.set_SUBDATA_LENGTH(newsgroups_data_len)

        if Configuration.get_SUMMERIZED():
            for i in range(newsgroups_data_len):
                document = newsgroups_data_document[i].replace("\n"," ").replace("\t", " ")
                parser = PlaintextParser.from_string(document, S_Tokenizer(Configuration.get_LANGUAGE()))
                stemmer = Stemmer(Configuration.get_LANGUAGE())
                summerizer = Summerizer(stemmer)
                summerizer.stop_words = get_stop_words(Configuration.get_LANGUAGE())
                document = ""
                for sentence in summerizer(parser.document, Configuration.get_SENTENCES_COUNT()):
                    document += str(sentence) + " "
                # data[i] = document
                data.append(document)

            data = data[:Configuration.get_SUBDATA_LENGTH()]
            newsgroups_data_target = newsgroups_data_target[:Configuration.get_SUBDATA_LENGTH()]

            for i in range(Configuration.get_SUBDATA_LENGTH()):
                for j in range(i, Configuration.get_SUBDATA_LENGTH()):
                    if newsgroups_data_target[i] == newsgroups_data_target[j]:
                        data_pair[str(i) + "_" + str(j)] = 1
                    else:
                        data_pair[str(i) + "_" + str(j)] = 0

            # for i in range(newsgroups_data_len):
            #     for j in range(i, newsgroups_data_len):
            #         if newsgroups_data_target[i] == newsgroups_data_target[j]:
            #             data_pair[str(i) + "_" + str(j)] = 1
            #         else:
            #             data_pair[str(i) + "_" + str(j)] = 0
        else:

            for i in range(newsgroups_data_len):
                # data[i] = newsgroups_data_document[i].replace("\n"," ").replace("\t", " ")
                data.append(newsgroups_data_document[i].replace("\n"," ").replace("\t", " "))

            data = data[:Configuration.get_SUBDATA_LENGTH()]
            newsgroups_data_target = newsgroups_data_target[:Configuration.get_SUBDATA_LENGTH()]

            for i in range(Configuration.get_SUBDATA_LENGTH()):
                for j in range(i, Configuration.get_SUBDATA_LENGTH()):
                    if newsgroups_data_target[i] == newsgroups_data_target[j]:
                        data_pair[str(i) + "_" + str(j)] = 1
                    else:
                        data_pair[str(i) + "_" + str(j)] = 0

            # for i in range(newsgroups_data_len):
            #     for j in range(i, newsgroups_data_len):
            #         if newsgroups_data_target[i] == newsgroups_data_target[j]:
            #             data_pair[str(i) + "_" + str(j)] = 1
            #         else:
            #             data_pair[str(i) + "_" + str(j)] = 0

        out = open(pkl_file, "wb")
        pickle.dump(data, out)
        out.close()
        out = open(pkl_file_pair, "wb")
        pickle.dump(data_pair, out)
        out.close()
    else:
        pkl_data = open(pkl_file, "rb")
        data = pickle.load(pkl_data)
        pkl_data.close()
        pkl_data_pair = open(pkl_file_pair, "rb")
        data_pair = pickle.load(pkl_data_pair)
        pkl_data_pair.close()
    return data, data_pair

def corpus():
    print("corpus....................................................")
    newsgroups_data = fetch_20newsgroups(subset="all")
    if not os.path.exists(Configuration.get_TEXT_DATA_SUMMERIZED_DIR()):
        os.mkdir(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR())

    data = {}
    no_of_warnings = 0
    if Configuration.get_SUMMERIZED():
        pkl_file = os.path.join(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR(),
                                str(Configuration.get_SENTENCES_COUNT()) +  "_sumy.pkl")
    else:
        pkl_file = os.path.join(Configuration.get_TEXT_DATA_SUMMERIZED_PICKLE_DIR(), "original.pkl")

    if not os.path.exists(pkl_file):
        newsgroups_data_document = newsgroups_data.data
        newsgroups_data_target = newsgroups_data.target
        newsgroups_data_len = len(newsgroups_data_document)

        print(newsgroups_data_len)

        if Configuration.get_SUMMERIZED():
            for i in range(newsgroups_data_len):
                document = newsgroups_data_document[i].replace("\n"," ").replace("\t", " ")
                parser = PlaintextParser.from_string(document, S_Tokenizer(Configuration.get_LANGUAGE()))
                stemmer = Stemmer(Configuration.get_LANGUAGE())
                summerizer = Summerizer(stemmer)
                summerizer.stop_words = get_stop_words(Configuration.get_LANGUAGE())
                document = ''
                for sentence in summerizer(parser.document, Configuration.get_SENTENCES_COUNT()):
                    document + str(sentence)
                data[document] = newsgroups_data_target[i]
        else:
            for i in range(newsgroups_data_len):
                document = newsgroups_data_document[i].replace("\n", " ").replace("\t", " ")
                data[document] = newsgroups_data_target[i]
        out = open(pkl_file, "wb")
        pickle.dump(data, out)
        out.close()
    else:
        pkl_data = open(pkl_file, "rb")
        data = pickle.load(pkl_data)
        pkl_data.close()
    return data, newsgroups_data.target_names



def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str)
    return m.hexdigest()