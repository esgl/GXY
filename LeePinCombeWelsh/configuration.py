from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
class Configuration:
	DataDir = '/data/LeePincombeWelsh'
	File_Similarity = DataDir + "/LeePincombeWelshData.csv"
	Document = DataDir + "/LeePincombeWelshDocuments.txt"
	GLOVE_DIR = '/data/glove.6B'
	gloveFile = GLOVE_DIR + "/glove.6B.100d.txt"
	MAX_SEQUENCE_LENGTH = 1000
	MAX_NB_WORDS = 20000
	MAX_SENTENCE_LENGTH_IN_DOCUMENT = 1
	EMBEDDING_DIM = 50
	VALIDATION_SPLIT = 0.2
	EPIDOSE = 1
	batch_size = 32
	pearson_thershold = 0.2
	weight = 1.2
	LOG = "/logs/leePincombeWelsh/results"

def write_configuration(logfile):
	f = open(logfile, mode="a+")
	f.write("...........................................................\n")
	f.write("DataDir = %s\n" % os.getcwd() + Configuration.DataDir)
	f.write("File_Similarity = %s\n" % os.getcwd() + Configuration.File_Similarity)
	f.write("Document = %s\n" % os.getcwd() + Configuration.Document)
	f.write("GLOVE_DIR = %s\n" % os.getcwd() + Configuration.GLOVE_DIR)
	f.write("gloveFile = %s\n" % os.getcwd() + Configuration.gloveFile)
	f.write("MAX_SEQUENCE_LENGTH %d\n" % Configuration.MAX_SEQUENCE_LENGTH)
	f.write("MAX_NB_WORDS %d\n" % Configuration.MAX_NB_WORDS)
	f.write("MAX_SENTENCE_LENGTH_IN_DOCUMENT %d\n" % Configuration.MAX_SENTENCE_LENGTH_IN_DOCUMENT)
	f.write("EMBEDDING_DIM = %d\n" % Configuration.EMBEDDING_DIM)
	f.write("VALIDATION_SPLIT = %d\n" % Configuration.VALIDATION_SPLIT)
	f.write("EPIDOSE = %d\n" % Configuration.EPIDOSE)
	f.write("BATCH_SIZE = %d\n" % Configuration.batch_size)
	f.write("PEARSON_THERSHOLD = %d\n" % Configuration.pearson_thershold)
	f.write("WEIGHT = %d\n" % Configuration.weight)
	f.close()


def set_LOG(logfile):
    Configuration.LOG = logfile
def get_LOG():
    return os.getcwd() + Configuration.LOG

def set_MAX_SENTENCE_LENGTH_IN_DOCUMENT(max_sentence_length_in_document):
	Configuration.MAX_SENTENCE_LENGTH_IN_DOCUMENT = max_sentence_length_in_document
def get_MAX_SENTENCE_LENGTH_IN_DOCUMENT():
	return Configuration.MAX_SENTENCE_LENGTH_IN_DOCUMENT


def set_Data_Dir(data_dir):
	Configuration.DataDir = data_dir
def get_Data_Dir():
	return os.getcwd() + Configuration.DataDir

def set_File_Similarity(file_similarity):
	Configuration.File_Similarity = file_similarity
def get_File_Similarity():
	return os.getcwd() + Configuration.File_Similarity

def set_Document(Document):
	Configuration.Document = Document
def get_Document():
	return os.getcwd() + Configuration.Document

def set_GLOVE_DIR(glove_dir):
	Configuration.GLOVE_DIR = glove_dir
def get_GLOVE_DIR():
	return os.getcwd() + Configuration.GLOVE_DIR

def set_GLOVEFILE(glove_file):
	Configuration.gloveFile = glove_file
def get_GLOVEFILE():
	return os.getcwd() + Configuration.gloveFile

def set_MAX_SEQUENCE_LENGTH(max_sequence_length):
	Configuration.MAX_SEQUENCE_LENGTH = max_sequence_length
def get_MAX_SEQUENCE_LENGTH():
	return Configuration.MAX_SEQUENCE_LENGTH

def set_MAX_NB_WORDS(max_nb_words):
	Configuration.MAX_NB_WORDS = max_nb_words
def get_MAX_NB_WORDS():
	return Configuration.MAX_NB_WORDS

def set_EMBEDDING_DIM(embedding_dim):
	Configuration.EMBEDDING_DIM = embedding_dim
def get_EMBEDDING_DIM():
	return Configuration.EMBEDDING_DIM

def set_VALIDATION_SPLIT(validation_split):
	Configuration.VALIDATION_SPLIT = validation_split
def get_VALIDATION_SPLIT():
	return Configuration.VALIDATION_SPLIT

def set_EPIDOSE(epidose):
	Configuration.EPIDOSE = epidose
def get_EPIDOSE():
	return Configuration.EPIDOSE

def set_BATCH_SIZE(batch_size):
	Configuration.batch_size = batch_size
def get_BATCH_SIZE():
	return Configuration.batch_size

def set_PEARSON_THERSHOLD(pearson_thershold):
	Configuration.pearson_thershold = pearson_thershold
def get_PEARSON_THRESHOLD():
	return Configuration.pearson_thershold

def set_WEIGHT(weight):
	Configuration.weight = weight
def get_WEIGTH():
	return Configuration.weight