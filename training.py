#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Package which train the neural network model,
can be launched from the command line
"""


from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from amazon_reviews.document import AmazonReviewParser, Vectorizer
from amazon_reviews.neural_network.recurrent import RecurrentNeuralNetwork


def _main() -> None:
    """
    Main function DO NOT IMPORT
    """
    experiment_name = 'base_professor_model'
    print('Reading training data')
    documents = AmazonReviewParser.read_file('Automotive_5_train.json')
    print('Create features')
    vectorizer = Vectorizer('glove.6B.50d.txt')
    word, pos, shape = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    print(f'Loaded {len(word)} data samples', '\n', 'Train...')
    input_shape = {
        'pos': (len(vectorizer.pos2index), 10),
        'shape': (len(vectorizer.shape2index), 2)
    }
    model = RecurrentNeuralNetwork.build_classification(vectorizer.word_embeddings, input_shape, 1)
    trained_model_name = './models_save/professor_ner_weights.h5'
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    save_best_model = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='auto')
    tb_callbacks = TensorBoard(f'./tf_logs/{experiment_name}')
    model.fit([word, pos, shape], labels, validation_split=0.2, batch_size=64,
              epochs=10, callbacks=[save_best_model, early_stopping, tb_callbacks])


if __name__ == '__main__':
    _main()
