from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

class Configuration:

    BASE_DATA_DIR = '/data'
    GLOVE_DIR = BASE_DATA_DIR + '/glove.6B/'
    GLOVE_FILE = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')

    TEXT_DATA_DIR = BASE_DATA_DIR + '/20_newsgroup/'
    TEXT_DATA_SUMMERIZED_DIR = TEXT_DATA_DIR + '20_newsgroup_summerized/'
    TEXT_DATA_SUMMERIZED_PICKLE_DIR = TEXT_DATA_SUMMERIZED_DIR + 'pickle/'

    EMBEDDING_INDEX_FILE = os.path.join(TEXT_DATA_SUMMERIZED_PICKLE_DIR, 'embedding_index.pkl')
    SUBDATA_LENGTH = 20
    LANGUAGE = "english"
    SENTENCES_COUNT = 5
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 32
    EPISODE = 50
    SUMMERIZED = True

    LOG = "/logs/20_newsgroup/results"

def write_configuration(logfile):
    f = open(logfile, mode="a+")
    f.write("...........................................................\n")
    f.write("GLOVE_DIR = %s\n" % os.getcwd() + Configuration.GLOVE_DIR)
    f.write("gloveFile = %s\n" % os.getcwd() + Configuration.GLOVE_FILE)
    f.write("TEXT_DATA_SUMMERIZED_PICKLE_DIR = %s\n" % os.getcwd() + Configuration.TEXT_DATA_SUMMERIZED_PICKLE_DIR)
    f.write("EMBEDDING_INDEX_FILE = %s\n" % os.getcwd() + Configuration.EMBEDDING_INDEX_FILE)
    f.write("SUBDATA_LENGTH = %d\n" % Configuration.SUBDATA_LENGTH)
    f.write("LANGUAGE = %s\n" % Configuration.LANGUAGE)
    f.write("SENTENCES_COUNT = %s\n" % Configuration.SENTENCES_COUNT)
    f.write("MAX_SEQUENCE_LENGTH %d\n" % Configuration.MAX_SEQUENCE_LENGTH)
    f.write("MAX_NB_WORDS %d\n" % Configuration.MAX_NB_WORDS)
    f.write("EMBEDDING_DIM = %d\n" % Configuration.EMBEDDING_DIM)
    f.write("VALIDATION_SPLIT = %d\n" % Configuration.VALIDATION_SPLIT)
    f.write("EPIDOSE = %d\n" % Configuration.EPISODE)
    f.write("BATCH_SIZE = %d\n" % Configuration.BATCH_SIZE)
    f.write("SUMMERIZED = %s\n" % "True" if Configuration.SUMMERIZED else "False")
    f.close()

def get_SUBDATA_LENGTH():
    return Configuration.SUBDATA_LENGTH
def set_SUBDATA_LENGTH(subData_Lenght):
    Configuration.SUBDATA_LENGTH = subData_Lenght

def get_LOG():
    return os.getcwd() + Configuration.LOG
def set_LOG(logfile):
    Configuration.LOG = logfile

def get_TEXT_DATA_SUMMERIZED_PICKLE_DIR():
    return os.getcwd() + Configuration.TEXT_DATA_SUMMERIZED_PICKLE_DIR
def set_TEXT_DATA_SUMMERIZED_PICKLE_DIR(text_data_summerized_pickle_dir):
    Configuration.TEXT_DATA_SUMMERIZED_PICKLE_DIR = text_data_summerized_pickle_dir

def get_TEXT_DATA_SUMMERIZED_DIR():
    return os.getcwd() + Configuration.TEXT_DATA_SUMMERIZED_DIR
def set_TEXT_DATA_SUMMERIZED_DIR(text_data_summerized_dir):
    Configuration.TEXT_DATA_SUMMERIZED_DIR = text_data_summerized_dir

def get_TEXT_DATA_DIR():
    return os.getcwd() + Configuration.TEXT_DATA_DIR
def set_TEXT_DATA_DIR(text_data_dir):
    Configuration.TEXT_DATA_DIR = text_data_dir

def get_SUMMERIZED():
    return Configuration.SUMMERIZED
def set_SUMMERIZED(summerized):
    Configuration.SUMMERIZED = summerized

def get_EPISODE():
    return Configuration.EPISODE
def set_EPISODE(episode):
    Configuration.EPISODE = episode

def get_BATCH_SIZE():
    return Configuration.BATCH_SIZE
def set_BATCH_SIZE(batch_size):
    Configuration.BATCH_SIZE = batch_size
def get_VALIDATION_SPLIT():
    return Configuration.VALIDATION_SPLIT
def set_VALIDATION_SPLIT(validation_split):
    Configuration.VALIDATION_SPLIT = validation_split

def get_EMBEDDING_DIM():
    return Configuration.EMBEDDING_DIM
def set_EMBEDDING_DIM(embedding_dim):
    Configuration.EMBEDDING_DIM = embedding_dim

def get_MAX_NB_WORDS():
    return Configuration.MAX_NB_WORDS
def set_MAX_NB_WORDS(max_nb_words):
    Configuration.MAX_NB_WORDS = max_nb_words

def get_MAX_SEQUENCE_LENGTH():
    return Configuration.MAX_SEQUENCE_LENGTH
def set_MAX_SEQUENCE_LENGTH(max_sequence_length):
    Configuration.MAX_SEQUENCE_LENGTH = max_sequence_length

def get_SENTENCES_COUNT():
    return Configuration.SENTENCES_COUNT
def set_SENTENCES_COUNT(sentences_count):
    Configuration.SENTENCES_COUNT = sentences_count

def get_LANGUAGE():
    return Configuration.LANGUAGE
def set_LANGUAGE(language):
    Configuration.LANGUAGE = language

def get_GLOVE_DIR():
    return os.getcwd() + Configuration.GLOVE_DIR
def set_GLOVE_DIR(glove_dir):
    Configuration.GLOVE_DIR = glove_dir

def get_GLOVE_FILE():
    return  os.getcwd() + Configuration.GLOVE_FILE
def set_GLOVE_FILE(glove_file):
    Configuration.GLOVE_FILE =  glove_file

def get_EMBEDDING_INDEX_FILE():
    return os.getcwd() + Configuration.EMBEDDING_INDEX_FILE
def set_EMBEDDING_INDEX_FILE(embedding_index_file):
    Configuration.EMBEDDING_INDEX_FILE = embedding_index_file
