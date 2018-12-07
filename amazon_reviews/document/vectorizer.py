#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
File for for vectorizing documents
"""


import os
from typing import List, Tuple
import numpy as np
from gensim.models import KeyedVectors
from config import GLOVE_DIR


class Vectorizer:
    """
    Transform a string into a vector representation
    """

    def __init__(self, word_embedding_path: str) -> None:
        """
        initialize the class
        :param word_embedding_path: path to gensim embedding file
        """
        filename = os.path.join(GLOVE_DIR, word_embedding_path)
        self.word_embeddings = KeyedVectors.load_word2vec_format(filename, binary=False)
        self.pos2index = {'PAD': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8,
                          '--': 9, 'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16,
                          'PRP$': 17, 'WDT': 18, '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25,
                          'RBR': 26, 'RBS': 27, 'VBD': 28, 'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33,
                          'PDT': 34, 'MD': 35, 'VB': 36, 'WRB': 37, 'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41,
                          'CC': 42, 'CD': 43, 'POS': 44, 'LS': 45}
        self.shape2index = {'NL': 0, 'NUMBER': 1, 'SPECIAL': 2, 'ALL-CAPS': 3,
                            '1ST-CAP': 4, 'LOWER': 5, 'MISC': 6}
        self.labels2index = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

    def encode_features(self, documents: List['amazon_reviews.document.Document'])\
            -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
        """
        TODO: Use numpy array instead of list
        Creates a feature matrix for all documents in the sample list
        :param documents: list of all samples as document objects
        :return: lists of numpy arrays for word, pos and shape features.
                 Each item in the list is a sentence, i.e. a list of indices (one per token)
        """
        words, pos, shapes = [], [], []
        for doc in documents:
            w, p, s = [], [], []
            for token in doc.tokens:
                if token.text.lower() in self.word_embeddings.index2word:
                    w.append(self.word_embeddings.index2word.index(token.text.lower()))
                else:
                    w.append(0)
                p.append(self.pos2index[token.pos])
                s.append(self.shape2index[token.shape])
            words.append(w)
            pos.append(p)
            shapes.append(s)
        return np.asarray(words, dtype=np.int32), np.asarray(pos, dtype=np.int8), np.asarray(shapes, dtype=np.int8)

    def encode_annotations(self, documents: List['amazon_reviews.document.Document']) -> 'np.ndarray':
        """
        Creates the Y matrix representing the annotations (or true positives) of a list of documents
        :param documents: list of documents to be converted in annotations vector
        :return: A numpy array where each item in the list is a sentence, i.e. a list of labels (one per token)
        """
        return np.asarray([self.labels2index[doc.rating] for doc in documents], dtype=np.int8)
