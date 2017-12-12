from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import LSTM, Input, Dense, Masking, Embedding, Activation, Concatenate, ThresholdedReLU
import numpy as np
from model.ThresholdedBinary import ThresholdedBinary

class ModelNewsGroup():
    def __init__(self, inputLength, embedding_weights):
        print("init................................................")
        input_document_a = Input(shape=(inputLength, ), name="input_document_a")
        input_document_b = Input(shape=(inputLength, ), name="input_document_b")
        embedding_a = Embedding(input_dim=embedding_weights.shape[0], output_dim=embedding_weights.shape[1],
                                input_length=inputLength, name="embedding_a", weights=[embedding_weights],
                                trainable=False)(input_document_a)
        embedding_b = Embedding(input_dim=embedding_weights.shape[0], output_dim=embedding_weights.shape[1],
                                input_length=inputLength, name="embedding_b", weights=[embedding_weights],
                                trainable=False)(input_document_b)
        lstm_a = LSTM(units=embedding_weights.shape[1], dropout=0.2, name="lstm_a")(embedding_a)
        lstm_b = LSTM(units=embedding_weights.shape[1], dropout=0.2, name="lstm_b")(embedding_b)
        concate = Concatenate(input_shape=(embedding_weights.shape[1], ), name="concate")([lstm_a, lstm_b])

        dense = Dense(units=inputLength, input_shape=(2 * inputLength, ), name="dense1")(concate)
        activation = Activation(input_shape=(inputLength, ), activation="sigmoid", name="activation")(dense)
        dense2 = Dense(units=1, input_shape=(inputLength, ), activation="softmax", name="dense2")(activation)
        # output = ThresholdedBinary(name="output")(dense2)
        output = ThresholdedReLU(theta=0.5, name="output")(dense2)
        self.model = Model(inputs=[input_document_a, input_document_b], outputs=output)

    def train(self, data_x_a, data_x_b, data_y, episode, batch_size):
        print("train...............................................")

        self.model.compile(optimizer="sgd", loss="mean_squared_error")
        # self.model.compile(optimizer="sgd", loss="binary_crossentropy")
        self.model.fit([data_x_a, data_x_b], data_y, epochs=episode, batch_size=batch_size)

    def predict(self, data_x_a, data_x_b, batch_size):
        print("predict.............................................")
        prediction = self.model.predict([data_x_a, data_x_b], batch_size=batch_size)
        prediction = list(map(lambda x: 1 if x > 0.5 else 0, prediction))
        return prediction

    def evaluation(self, data_x_a, data_x_b, data_y, batch_size):
        prediction = self.model.predict([data_x_a, data_x_b], batch_size=batch_size)
        prediction = list(map(lambda x: 1 if x > 0.5 else 0, prediction))
        length = len(data_y)
        count = 0
        for i in range(length):
            if prediction[i] == data_y[i]:
                count += 1
        precise = float(count) / length
        return precise

