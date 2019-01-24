#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the neural_network/recurrent.py file
Wrapper functions are not tested
"""


from keras.models import Model
import numpy as np
import pytest

from amazon_reviews.document import Vectorizer
from amazon_reviews.neural_network.recurrent import RecurrentNeuralNetwork


def test_RecurrentNeuralNetwork_build_classification() -> None:
    """
    Test The build_classification class method
    """
    vectorizer = Vectorizer('glove.6B.50d.txt')
    input_shape = {
        'pos': (len(vectorizer.pos2index), 10),
        'shape': (len(vectorizer.shape2index), 2)
    }
    rnn = RecurrentNeuralNetwork.build_classification(vectorizer.word_embeddings, input_shape, 1)
    assert isinstance(rnn._model, Model)


def test_RecurrentNeuralNetwork_probas_to_classes():
    """
    Test The probas_to_classes class method
    """
    arr1 = np.asarray([0.1, 0.2, 0.7], dtype=np.float32)
    arr2 = np.asarray([0.1], dtype=np.float32)
    assert RecurrentNeuralNetwork.probas_to_classes(arr1) == 2
    assert RecurrentNeuralNetwork.probas_to_classes(arr2) == 0
