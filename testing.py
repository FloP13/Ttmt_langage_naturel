#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Test the neural network model
"""


from pprint import pprint
import numpy as np
from amazon_reviews.document import AmazonReviewParser, Vectorizer
from amazon_reviews.neural_network.recurrent import RecurrentNeuralNetwork


def main() -> None:
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
    predicted = []
    for fi in range(len(word)):
        pred = model.predict([word[fi], pos[fi], shape[fi]], batch_size=1, verbose=0)
        pred_class = RecurrentNeuralNetwork.probas_to_classes(pred)
        predicted.append(pred_class)
    accuracy = sum([1 for p, l in zip(predicted, labels) if p == l]) / nb_features
    print(f'Accuracy of : {accuracy}%')


if __name__ == '__main__':
    main()
