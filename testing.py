#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Test the neural network model, can be launched from the command line prompt
"""


import numpy as np

from amazon_reviews.document import AmazonReviewParser, Vectorizer
from amazon_reviews.neural_network.recurrent import RecurrentNeuralNetwork


def _is_rating_concordant_comment(comment: int, rating: int) -> bool:
    """
    Check if a rating and a comment are concordant
    :param comment: The status of a comment (negative: 0, positive: 1)
    :param rating: The rating of the document
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
    ratings = (doc.rating for doc in documents)
    nb_features = len(word)
    print(f'Loaded {nb_features} data samples', '\n', 'Predicting...')
    model = RecurrentNeuralNetwork.load('./models_save/ner_weights.h5')
    predicted = model.predict([word, pos, shape], batch_size=64)
    predicted_classes = np.asarray([RecurrentNeuralNetwork.probas_to_classes(p) for p in predicted], dtype=np.int8)
    accuracy = sum((1 for p, l in zip(predicted_classes, labels) if p == l)) / nb_features
    accuracy_ratings_comments = sum((1 for p, r in zip(predicted_classes, ratings)
                                     if _is_rating_concordant_comment(p, r))) / nb_features
    print(f'Prediction accuracy of : {accuracy*100}%')
    print(f'Predicted accuracy of ratings and concordant comments {accuracy_ratings_comments*100}%')


if __name__ == '__main__':
    _main()
