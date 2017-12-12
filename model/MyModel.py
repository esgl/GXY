from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import LSTM, Input, Dense, Masking, Embedding

from model.CosineLayer import CosineLayer

import numpy as np

class MyModel:
    def __init__(self, inputLenght, weights):
        input_document_a = Input(shape=(inputLenght, ), name="input_document_a")
        input_document_b = Input(shape=(inputLenght, ), name="input_document_b")

        embedding_a = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                input_length=inputLenght, name="embedding_a",
                                weights=[weights], trainable=False)(input_document_a)

        embedding_b = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                input_length=inputLenght, name="embedding_b",
                                weights=[weights], trainable=False)(input_document_b)
        lstm_a = LSTM(units=weights.shape[1], dropout=0.2 ,name="lstm_a")(embedding_a)
        lstm_b = LSTM(units=weights.shape[1], dropout=0.2, name="lstm_b")(embedding_b)

        output = CosineLayer(output_dim=(None, 1), name="cosine")([lstm_a, lstm_b])

        self.model = Model(inputs=[input_document_a, input_document_b], outputs=output)

    def train(self, data_x_a, data_x_b, data_y, episode, batch_size):
        print("train..........................")
        self.model.compile(optimizer="sgd", loss="mean_squared_error")
        self.model.fit([data_x_a, data_x_b], data_y, epochs=episode,
                       batch_size=batch_size, verbose=1)

    def predict(self, data_x_a, data_x_b, batch_size):
        print("predict..........................")
        return self.model.predict([data_x_a, data_x_b], batch_size=batch_size)

    def evaluate(self, data_x_a, data_x_b, data_y, batch_size):
        # print(data_y)
        print("evaluate.........................")
        prediction = self.model.predict([data_x_a, data_x_b], batch_size=batch_size)

        cosine = np.matmul(prediction, np.transpose(data_y)) / \
            (np.sqrt(np.matmul(prediction, np.transpose(prediction)))*
             np.sqrt(np.matmul(data_y, np.transpose(data_y))))
        pearson = (len(prediction) * np.matmul(prediction, np.transpose(data_y)) - np.sum(prediction) * np.sum(data_y)) / \
                  ((np.sqrt(len(prediction) * np.matmul(prediction, np.transpose(prediction)) - np.power(np.sum(prediction), 2)))* \
                   (np.sqrt(len(data_y) * np.matmul(data_y, np.transpose(data_y)) - np.power(np.sum(data_y), 2))))
        print(type(cosine))
        print(type(pearson))


        return cosine, pearson

    def get_some_result(self, data_x_a, data_x_b):
        model_some = Model(self.model.inputs, self.model.get_layer(name='cosine').output)
        print(model_some.predict([data_x_a, data_x_b]))