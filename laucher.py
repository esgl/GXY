from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
# import time as t
import time
import LeePinCombeWelsh.configuration as Configuration
import newsgroup20.configuration as Configuration_NewsGroup

from model.MyModel import MyModel
from model.ModelNewsGroup import  ModelNewsGroup

from LeePinCombeWelsh.generate_data import generate_data as generate_data_leePincombeWelsh
from newsgroup20.generate_data import generate_data as generate_data_20newsgroup

t = time.strftime("%Y-%m-%d %H:%I:%S", time.localtime( time.time() ) )


def main_20newsgroup():
    f = open(Configuration_NewsGroup.get_LOG(), "a+")
    f.write("\n\n......................Experiment Begin......................\n")
    f.write("%s\n" % t)
    f.close()
    Configuration_NewsGroup.write_configuration(logfile=Configuration_NewsGroup.get_LOG())

    (document_a_train, document_b_train, category_train, document_a_test, document_b_test, category_test), embedding_weights_ = generate_data_20newsgroup()
    model = ModelNewsGroup(inputLength=Configuration.get_MAX_SEQUENCE_LENGTH(), embedding_weights=embedding_weights_)
    model.train(data_x_a=document_a_train, data_x_b=document_b_train, data_y=category_train,
                episode=Configuration_NewsGroup.get_EPISODE(), batch_size=Configuration_NewsGroup.get_BATCH_SIZE())
    f = open(Configuration_NewsGroup.get_LOG(), "a+")
    preciseTrain = model.evaluation(data_x_a=document_a_train, data_x_b=document_b_train, data_y=category_train,
                                    batch_size=Configuration_NewsGroup.get_BATCH_SIZE())
    print("precise of train data = %f" % preciseTrain)
    f.write("precise of train data = %f\n" % preciseTrain)

    preciseTest = model.evaluation(data_x_a=document_a_test, data_x_b=document_b_test, data_y=category_test,
                                   batch_size=Configuration_NewsGroup.get_BATCH_SIZE())
    print("precise of test data = %f" % preciseTrain)
    f.write("precise of test data = %f\n" % preciseTrain)
    f.close()
    f = open(Configuration_NewsGroup.get_LOG(), mode="a+")
    f.write("......................Experiment End......................\n\n")
    f.close()


def main_leePincombeWelsh():

    f = open(Configuration.get_LOG(), "a+")
    f.write("\n\n......................Experiment Begin......................\n")
    f.write("%s\n" % t.time())
    f.close()
    Configuration.write_configuration(logfile=Configuration.get_LOG())

    data_x_a, data_x_b, data_y, embedding_matrix = generate_data_leePincombeWelsh()
    print(data_x_a.shape)
    print(data_x_a[1:3,:])
    print(data_x_b.shape)
    print(data_y.shape)
    print(embedding_matrix.shape)
    print(embedding_matrix[1:3,:])
    # exit()

    indices = np.arange(data_x_a.shape[0])
    np.random.shuffle(indices)

    data_x_a = data_x_a[indices]
    data_x_b = data_x_b[indices]
    data_y = data_y[indices]

    nb_validation_samples = int(Configuration.get_VALIDATION_SPLIT() * data_x_a.shape[0])
    x_a_train = data_x_a[:-nb_validation_samples]
    x_b_train = data_x_b[:-nb_validation_samples]
    y_train = data_y[:-nb_validation_samples]

    x_a_test = data_x_a[-nb_validation_samples:]
    x_b_test = data_x_b[-nb_validation_samples:]
    y_test = data_y[-nb_validation_samples:]

    # model = MyModel_New(embedding_matrix)
    model = MyModel(inputLenght=Configuration.get_MAX_SEQUENCE_LENGTH(), weights=embedding_matrix)
    model.train(data_x_a=x_a_train, data_x_b=x_b_train, data_y=y_train,
                batch_size=Configuration.get_BATCH_SIZE(), episode=Configuration.get_EPIDOSE())

    # print(model.predict(x_a_test, x_b_test))
    f = open(Configuration.get_LOG(), mode="a+")
    cosTrain, pearsonTrain = model.evaluate(data_x_a=x_a_train, data_x_b=x_b_train, data_y=y_train,
                              batch_size=Configuration.get_BATCH_SIZE())
    print("cosine of train data = %f, pearson of train data = %f" % (cosTrain, pearsonTrain))
    f.write("cosine of prediction of train data = %f\n" % cosTrain)

    cosTest, pearsonTest = model.evaluate(data_x_a=x_a_test, data_x_b=x_b_test, data_y=y_test, batch_size=Configuration.get_BATCH_SIZE())
    f.write("cosine of prediction of test data = %f\n" % cosTest)
    print("cosine of test data = %f, pearson of test data = %f" % (cosTest, pearsonTest))
    f.close()

    f = open(Configuration.get_LOG(), mode="a+")
    f.write("......................Experiment End......................\n\n")
    f.close()


if __name__ == "__main__":
    # dataset = 1 # LeePincombeWelsh
    dataset = 2  # 20 newsgroup
    if dataset == 1:
        main_leePincombeWelsh()
    elif dataset == 2:
        main_20newsgroup()

