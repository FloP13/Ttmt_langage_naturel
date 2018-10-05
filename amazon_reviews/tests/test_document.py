#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document.py file
TODO: Test pos_tag
"""


import pytest
from amazon_reviews.document import Document


@pytest.fixture
def test_text() -> str:
    """
    A standard text as use case
    :return: A test text
    """
    return """Hello world !"""


def test_Document() -> None:
    """
    Test everything about the Document class
    """
    doc = Document.create_from_text(test_text())
    tokens = [token.text for token in doc.tokens]
    sentences = [(sentence.start, sentence.end) for sentence in doc.sentences]
    assert tokens == ['Hello', 'world', '!']
    assert sentences == [(0, 13)]
