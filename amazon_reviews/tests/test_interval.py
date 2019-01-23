#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document/interval.py file
TODO: Implement all the `pass` function
"""


import pytest

from amazon_reviews.document import *


@pytest.fixture
def test_text() -> str:
    """
    A standard text as use case
    :return: A test text
    """
    return """Hello world !
    I'm Thomas, nice to meet you."""


def test_Interval() -> None:
    """
    Test everything about the Interval class
    """
    interval = Interval(5, 10)
    assert interval.start == 5
    assert interval.end == 10
    assert len(interval) == 5
    assert Interval(5, 10) == Interval(5, 10)
    assert Interval(5, 10) >= Interval(5, 10)
    assert Interval(5, 10) <= Interval(5, 10)
    assert Interval(5, 10) < Interval(15, 20)
    assert Interval(15, 20) > Interval(5, 10)
    assert interval.start < 8 < interval.end
    assert str(interval) == 'Interval[5, 10]'
    intersect_interval = Interval(5, 10).intersection(Interval(8, 12))
    assert intersect_interval.start == 8 and intersect_interval.end == 10
    assert Interval(5, 10).overlaps(Interval(8, 12))
    interval.shift(10)
    assert interval.start == 15 and interval.end == 20


def test_Token() -> None:
    """
    Test everything about the Token class
    """
    token = Token(Document(), 0, 5, 'tt', 'tt', 'toto')
    # assert token.document == Document()
    assert token.start == 0
    assert token.end == 5
    assert token.pos == 'tt'
    assert token.shape == 'tt'
    assert token.text == 'toto'
    assert str(token) == 'Token(toto, 0, 5)'


def test_Sentence() -> None:
    """
    Test everything about the Token class
    """
    doc = Document.create_from_text(test_text())
    sentence = Sentence(doc, 0, 13)
    # assert sentence.document == doc
    assert sentence.start == 0
    assert sentence.end == 13
    assert str(sentence) == 'Sentence(0, 13)'
    tokens = [token.text for token in sentence.tokens]
    assert tokens == ['Hello', 'world', '!']


def test_get_shape_category() -> None:
    """
    Test the Document.get_shape_category function
    """
    pass
