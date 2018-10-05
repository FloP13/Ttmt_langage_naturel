#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Pytest file for the document.py file
TODO: Implement all the `pass` function
"""


import pytest


@pytest.fixture
def test_text() -> str:
    """
    A standard text as use case
    :return: A test text
    """
    return """Hello world !
    I'm Thomas, nice to meet you."""


def test_Document() -> None:
    """
    Test everything about the Token class
    """
    pass
