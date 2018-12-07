#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document/vectorizer.py file
"""


import pytest
from amazon_reviews.document import Vectorizer
from .test_document import document_setup


@pytest.fixture
def vectorizer_setup() -> Vectorizer:
    """
    Setup a vectorizer object for testing
    :return: A Vectorizer object
    """
    return Vectorizer('glove.6B.50d.txt')


def test_Vectorizer() -> None:
    """
    Test everything about the Vectorizer class
    """
    v = vectorizer_setup()
    docs = [document_setup()]
    annotations = v.encode_annotations(docs)
    words, pos, shapes = v.encode_features(docs)
    assert annotations.tolist() == [1]
    assert words.tolist() == [[13075, 85, 805]]
    assert pos.tolist() == [[38, 11, 21]]
    assert shapes.tolist() == [[4, 5, 2]]
