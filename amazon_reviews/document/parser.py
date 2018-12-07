#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
File for parsing documents
"""


import json
from typing import List
from .document import Document


class Parser:
    """
    Parent class for all parser
    """

    def read_file(self, filename: str) -> List[Document]:
        """
        Read a file and return a Document
        :param filename: The file path to load
        :return: The constructed Document
        """
        docs = []
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                d = self.read(line)
                docs.append(d)
        return docs

    def read(self, content: str) -> Document:
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

    def read(self, content: str) -> Document:
        """
        Read the content of the file and return a Document
        :param content: The content of the the text
        :return: The constructed Document
        """
        reviews = json.loads(content)
        doc = Document().create_from_text(reviews['reviewText'])
        doc.rating = reviews['overall']
        return doc
