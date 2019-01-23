#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document/document.py file
"""


import pytest

from amazon_reviews.document import Document


@pytest.fixture
def document_setup() -> Document:
    """
    A standard text as use case
    :return: A test text
    """
    txt = """Hello world !"""
    doc = Document.create_from_text(txt)
    doc.rating = 5
    return doc


def test_Document() -> None:
    """
    Test everything about the Document class
    """
    doc = document_setup()
    tokens = [token.text for token in doc.tokens]
    sentences = [(sentence.start, sentence.end) for sentence in doc.sentences]
    assert tokens == ['Hello', 'world', '!']
    assert sentences == [(0, 13)]
    assert doc.rating == 5.0
