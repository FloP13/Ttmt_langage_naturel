#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document/parser.py file
"""


import pytest
from amazon_reviews.document import AmazonReviewParser


@pytest.fixture
def test_review_file_path() -> str:
    """
    Get the path of the amazon review file
    :return: The path of the amazon review file
    """
    return '../amazon_reviews/tests/ressources/amazon_review_test.json'


def test_AmazonReviewParser() -> None:
    """
    Test the AmazonReviewParser Class
    """
    docs = AmazonReviewParser().read_file(test_review_file_path())
    assert docs[0].text == 'Doudoux le doux !'
    assert docs[0].rating == 5.0
    assert docs[1].text == 'Flo le déglingo !'
    assert docs[1].rating == 4.0
