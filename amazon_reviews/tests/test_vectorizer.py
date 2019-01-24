#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document/vectorizer.py file
"""


import pytest

from amazon_reviews.document import Vectorizer
from .test_document import document


@pytest.fixture
def vectorizer() -> Vectorizer:
    """
    Setup a vectorizer object for testing
    :return: A Vectorizer object
    """
    return Vectorizer('glove.6B.50d.txt')


@pytest.mark.usefixtures('document')
def test_Vectorizer(vectorizer: Vectorizer, document: 'amazon_reviews.document.Document') -> None:
    """
    Test everything about the Vectorizer class
    :param vectorizer: The fixture vectorizer to test on
    :param document: The fixture document to run test on
    """
    docs = [document]
    annotations = vectorizer.encode_annotations(docs)
    words, pos, shapes = vectorizer.encode_features(docs)
    assert annotations.tolist() == [1]
    assert words.tolist() == [[13075, 85, 805]]
    assert pos.tolist() == [[38, 11, 21]]
    assert shapes.tolist() == [[4, 5, 2]]
