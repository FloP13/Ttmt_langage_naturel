#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document/document.py file
"""


import pytest

from amazon_reviews.document import Document


@pytest.fixture
def document() -> Document:
    """
    A standard text as use case
    :return: A test text
    """
    txt = """Hello world !"""
    doc = Document.create_from_text(txt)
    doc.rating = 5
    return doc


def test_Document(document: Document) -> None:
    """
    Test everything about the Document class
    :param document: The fixture document to run test on
    """
    tokens = [token.text for token in document.tokens]
    sentences = [(sentence.start, sentence.end) for sentence in document.sentences]
    assert tokens == ['Hello', 'world', '!']
    assert sentences == [(0, 13)]
    assert document.rating == 5.0
