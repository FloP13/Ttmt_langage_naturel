#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Package which define recurrent Neural network models
"""


from keras.layers import concatenate, Input, Embedding, Dropout, Bidirectional, LSTM, Dense
from keras.models import Model, load_model


class RecurrentNeuralNetwork:
    """
    Wrapper class for managing Keras recurrent network usage
    """

    def __init__(self, model: Model = None) -> None:
        """
        Init the class with a model if provided else None
        :param model: The Keras `Model` to use
        """
        self._model = model

    def load_weights(self, *args, **kwargs) -> None:
        """
        Wrapper around `Model.load_weights`
        :param args: The args to pass to the underlying function
        :param kwargs: The kwargs to pass to the underlying function
        """
        self._model.load_weights(*args, **kwargs)

    def fit(self, *args, **kwargs) -> 'keras.callbacks.History':
        """
        Wrapper around `Model.fit`
        :param args: The args to pass to the underlying function
        :param kwargs: The kwargs to pass to the underlying function
        """
        return self._model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs) -> 'keras.callbacks.History':
        """
        Wrapper around `Model.fit_generator`
        :param args: The args to pass to the underlying function
        :param kwargs: The kwargs to pass to the underlying function
        """
        return self._model.fit_generator(*args, **kwargs)

    def predict_generator(self, *args, **kwargs) -> 'numpy.ndarray':
        """
        Wrapper around `Model.predict_generator`
        :param args: The args to pass to the underlying function
        :param kwargs: The kwargs to pass to the underlying function
        """
        return self._model.predict_generator(*args, **kwargs)

    def predict(self, *args, **kwargs) -> 'numpy.ndarray':
        """
        Wrapper around `Model.predict`
        :param args: The args to pass to the underlying function
        :param kwargs: The kwargs to pass to the underlying function
        """
        return self._model.predict(*args, **kwargs)

    @staticmethod
    def probas_to_classes(proba: 'numpy.ndarray') -> int:
        """
        Get the class with the highest probability
        :param proba: The vector of probability of classes
        :return: The ID of the class with the highest probability
        """
        return proba.argmax(axis=-1) if proba.shape[-1] > 1 else (proba > 0.5).astype('int32')

    @classmethod
    def build_classification(cls, word_embeddings: 'gensim.models.word2vec.Wod2Vec', input_shape: dict, out_shape: int,
                             units: int = 128, dropout_rate: float = 0.4) -> 'RecurrentNeuralNetwork':
        """
        Build the RNN classification models
        :param word_embeddings: Gensim Wod2Vec Model vector for word representation
        :param input_shape: The input shape of the model
        :param out_shape: The out shape of the model
        :param units: the number of unit for the model
        :param dropout_rate: The Dropout rate for the model
        :return: An initialized `RecurrentNeuralNetwork` object
        """
        print('Building RNN models')
        word_input = Input(shape=(None,), dtype='int32', name='word_input')
        weights = word_embeddings.syn0
        word_embeddings = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                    weights=[weights], name='word_embeddings_layer', trainable=False,
                                    mask_zero=True)(word_input)
        pos_input = Input(shape=(None,), dtype='int32', name='pos_input')
        pos_embeddings = Embedding(input_shape['pos'][0], input_shape['pos'][1], name='pos_embeddings_layer',
                                   mask_zero=True)(pos_input)
        shape_input = Input(shape=(None,), dtype='int32', name='shape_input')
        shape_embeddings = Embedding(input_shape['shape'][0], input_shape['shape'][1], name='shape_embeddings_layer',
                                     mask_zero=True)(shape_input)
        merged_input = concatenate([word_embeddings, pos_embeddings, shape_embeddings], axis=-1)
        bilstm = Bidirectional(LSTM(units, activation='tanh', return_sequences=True), name='bi-lstm')(merged_input)
        lstm = LSTM(units, activation='tanh', name='lstm')(bilstm)
        lstm_layer = Dropout(dropout_rate, name='second_dropout')(lstm)
        output = Dense(out_shape, activation='sigmoid', name='output')(lstm_layer)
        model = Model(inputs=[word_input, pos_input, shape_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return RecurrentNeuralNetwork(model)

    @classmethod
    def load(cls, filename: str) -> 'RecurrentNeuralNetwork':
        """
        Wrapper around `Model.load_model` to load a model
        :param filename: The filename to use for loading
        :return: An initialized `RecurrentNeuralNetwork` object
        """
        return RecurrentNeuralNetwork(load_model(filename))

    def save(self, filename: str) -> None:
        """
        Wrapper around `Model.save` to save a model
        :param filename: The filename to use for saving
        """
        self._model.save(filename)
