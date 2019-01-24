#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Package for parsing documents
"""


import json
import os
from typing import List, Optional

from config import DATA_DIR
from .document import Document


class Parser:
    """
    Parent class for all parser
    """

    @classmethod
    def read_file(cls, filename: str) -> List[Document]:
        """
        Read a file and return a Document
        :param filename: The file path to load
        :return: The constructed Document
        """
        docs = []
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                doc = cls.read(line)
                if doc is not None:
                    docs.append(doc)
        return docs

    @classmethod
    def read(cls, content: str) -> Document:
        """
        Read the content of the file and return a Document
        :param content: The content of the the text
        :return: The constructed Document
        """
        raise NotImplementedError


class AmazonReviewParser(Parser):
    """
    Class for parsing a Review from Amazon
    """

    @classmethod
    def read(cls, content: str) -> Optional[Document]:
        """
        Read the content of the file and return a Document if the doc is non empty
        :param content: The content of the the text
        :return: The constructed Document
        """
        reviews = json.loads(content)
        if reviews['reviewText'] and reviews['overall']:
            doc = Document.create_from_text(reviews['reviewText'])
            doc.rating = reviews['overall']
            return doc
        return None
