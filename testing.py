#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Package which test the neural network model,
can be launched from the command line
"""


import numpy as np
from sklearn.metrics import classification_report

from amazon_reviews.document import AmazonReviewParser, Vectorizer
from amazon_reviews.neural_network.recurrent import RecurrentNeuralNetwork


def _is_rating_concordant_comment(comment: int, rating: int) -> bool:
    """
    Check if a rating and a comment are concordant
    :param comment: The status of a comment (negative: 0, positive: 1)
    :param rating: The rating of the document (1 to 5)
    :return: If a rating and a comment are concordant
    """
    if rating <= 3 and comment == 0:
        return True
    if rating >= 4 and comment == 1:
        return True
    return False


def _main() -> None:
    """
    Main function DO NOT IMPORT
    """
    print('Reading Testing data')
    documents = AmazonReviewParser().read_file('Automotive_5_test.json')
    print('Create features')
    vectorizer = Vectorizer('glove.6B.50d.txt')
    word, pos, shape = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    nb_features = len(word)
    print(f'Loaded {nb_features} data samples', '\n', 'Predicting...')
    model = RecurrentNeuralNetwork.load('./models_save/ner_weights.h5')
    predicted = model.predict([word, pos, shape], batch_size=64)
    predicted_classes = np.asarray([RecurrentNeuralNetwork.probas_to_classes(p) for p in predicted], dtype=np.int8)
    print(classification_report(labels, predicted_classes, ['negative', 'positive']))


if __name__ == '__main__':
    _main()
