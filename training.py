#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Train the neural network model
"""


import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from amazon_reviews.document import AmazonReviewParser, Vectorizer
from amazon_reviews.neural_network.recurrent import RecurrentNeuralNetwork


def main() -> None:
    """
    Main function DO NOT IMPORT
    """
    print('Reading training data')
    documents = AmazonReviewParser().read_file('Automotive_5_train.json')
    print('Create features')
    vectorizer = Vectorizer('glove.6B.50d.txt')
    word, pos, shape = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    labels = [np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels2index)) for y_group in labels]
    labels = np.asarray(labels, dtype=np.float32)
    print(f'Loaded {len(word)} data samples', '\n', 'Train...')
    input_shape = {
        'pos': (len(vectorizer.pos2index), 10),
        'shape': (len(vectorizer.shape2index), 2)
    }
    out_shape = len(vectorizer.labels2index)
    model = RecurrentNeuralNetwork.build_classification(vectorizer.word_embeddings, input_shape, out_shape, 100, 0.5)
    trained_model_name = './models_save/ner_weights.h5'
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    save_best_model = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='auto')
    model.fit([word, pos, shape], labels, validation_split=0.8, batch_size=1024,
              epochs=20, callbacks=[save_best_model, early_stopping])


if __name__ == '__main__':
    main()
