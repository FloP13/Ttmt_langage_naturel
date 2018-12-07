#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
File for handling everything related to an Interval in a text document.
From sentences `Sentence` to words `Token`
"""


import re
from typing import List


class Interval:
    """
    A class for representing a contiguous range of integers
    """

    def __init__(self, start: int, end: int) -> None:
        """
        Constructor of the Interval class
        :param start: start of the range
        :param end: first integer not included the range
        """
        self._start = int(start)
        self._end = int(end)
        if self.start > self.end:
            raise ValueError('Start "{}" must not be greater than end "{}"'.format(self.start, self.end))
        if self.start < 0:
            raise ValueError('Start "{}" must not be negative'.format(self.start))

    def __len__(self) -> int:
        """
        Compute the len of the Interval object
        :return: end - start
        """
        return self.end - self.start

    def __eq__(self, other: 'Interval') -> bool:
        """
        Check if this interval and an other are equal
        :param other: The Interval object to be compared with
        :return: If this is equivalent or not
        """
        return self.start == other.start and self.end == other.end

    def __ne__(self, other: 'Interval') -> bool:
        """
        Check if this interval and an other are not equal
        :param other: The Interval object to be compared with
        :return: If this is not equivalent or not
        """
        return self.start != other.start or self.end != other.end

    def __lt__(self, other: 'Interval') -> bool:
        """
        Check if this interval is lesser than an other Interval
        :param other: The Interval object to be compared with
        :return: If this is lesser or not
        """
        return (self.start, -len(self)) < (other.start, -len(other))

    def __le__(self, other: 'Interval') -> bool:
        """
        Check if this interval is lesser or equal than an other Interval
        :param other: The Interval object to be compared with
        :return: If this is lesser or equal or not
        """
        return (self.start, -len(self)) <= (other.start, -len(other))

    def __gt__(self, other: 'Interval') -> bool:
        """
        Check if this interval is greater than an other Interval
        :param other: The Interval object to be compared with
        :return: If this is greater or not
        """
        return (self.start, -len(self)) > (other.start, -len(other))

    def __ge__(self, other: 'Interval') -> bool:
        """
        Check if this interval is greater or equal than an other Interval
        :param other: The Interval object to be compared with
        :return: If this is greater or equal or not
        """
        return (self.start, -len(self)) >= (other.start, -len(other))

    def __hash__(self) -> hash:
        """
        Compute the hash of the interval object
        :return: Hash of the interval object
        """
        return hash(tuple(v for k, v in sorted(self.__dict__.items())))

    def __contains__(self, item: int) -> bool:
        """
        Return self.start <= item < self.end
        :param item: The index of the item to be checked
        :return: If the element is in the Interval or not
        """
        return self.start <= item < self.end

    def __repr__(self) -> str:
        """
        The string representation of an Interval
        :return: The string representation of an Interval
        """
        return 'Interval[{}, {}]'.format(self.start, self.end)

    def __str__(self) -> str:
        """
        The string representation of an Interval
        :return: The string representation of an Interval
        """
        return repr(self)

    def intersection(self, other) -> 'Interval':
        """
        Return the interval common to self and other
        :param other: The interval to intersect with
        """
        a, b = sorted((self, other))
        if a.end <= b.start:
            return Interval(self.start, self.start)
        return Interval(b.start, min(a.end, b.end))

    def overlaps(self, other: 'Interval') -> bool:
        """
        Return True if there exists an interval common to self and other
        :param other: The interval to check the overlaps
        :return: If the is an interval or not as a bool
        """
        a, b = sorted((self, other))
        return a.end > b.start

    def shift(self, i: int) -> None:
        """
        Shift the interval
        :param i: The number of index to shift to
        """
        self._start += i
        self._end += i

    @property
    def start(self) -> int:
        """
        Start of token in interval
        :return: Start of token in interval
        """
        return self._start

    @property
    def end(self) -> int:
        """
        End of token in interval
        :return: end of token in interval
        """
        return self._end


class Token(Interval):
    """
    A Interval representing word like units of text with a dictionary of features
    """

    def __init__(self, document: 'Document', start: int, end: int, pos: str, shape: str, text: str) -> None:
        """
        Constructor of the Token class
        Note that a token has 2 text representations.
        1) How the text appears in the original document e.g. doc.text[token.start:token.end]
        2) How the tokenizer represents the token e.g. nltk.word_tokenize('"') == ['``']
        :param document: the document object containing the token
        :param start: start of token in document text
        :param end: end of token in document text
        :param pos: part of speach of the token
        :param shape: string label describing the shape of the token
        :param text: this is the text representation of token
        """
        Interval.__init__(self, start, end)
        self._doc = document
        self._start = int(start)
        self._end = int(end)
        self._pos = pos
        self._shape = shape
        self._text = text

    @property
    def text(self) -> str:
        """
        The text representation of token
        :return: the text representation of token
        """
        return self._text

    @property
    def pos(self) -> str:
        """
        Part of speach of the token
        :return: part of speach of the token
        """
        return self._pos

    @property
    def shape(self) -> str:
        """
        Integer label describing the shape of the token
        :return: integer label describing the shape of the token
        """
        return self._shape

    @property
    def start(self) -> int:
        """
        Start of token in document text
        :return: start of token in document text
        """
        return self._start

    @property
    def end(self) -> int:
        """
        End of token in document text
        :return: end of token in document text
        """
        return self._end

    @property
    def document(self) -> 'Document':
        """
        The document object containing the token
        :return: the document object containing the token
        """
        return self._doc

    def __repr__(self) -> str:
        """
        The representation of the Token
        :return: The representation of the Token as str
        """
        return 'Token({}, {}, {})'.format(self.text, self.start, self.end)

    def __str__(self) -> str:
        """
        The Token as string
        :return: the Token as str
        """
        return repr(self)


class Sentence(Interval):
    """
    Interval corresponding to a Sentence
    """

    def __init__(self, document: 'Document', start: int, end: int) -> None:
        """
        Constructor of the Sentence class
        :param document: the document object containing the token
        :param start: start of token in document text
        :param end: end of token in document text
        """
        Interval.__init__(self, start, end)
        self._doc = document
        self._start = int(start)
        self._end = int(end)

    def __repr__(self) -> str:
        """
        The representation of the Token
        :return: The representation of the Token as str
        """
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self) -> List['Token']:
        """
        Get list of tokens contained in a sentence
        :return: the list of tokens contained in a sentence
        """
        return [token for token in self._doc.tokens if self.overlaps(token)]

    @property
    def document(self) -> 'Document':
        """
        The document object containing the sentence
        :return: the document object containing the token
        """
        return self._doc

    @property
    def start(self) -> int:
        """
        Start of sentence in document text
        :return: start of sentence in document text
        """
        return self._start

    @property
    def end(self) -> int:
        """
        End of sentence in document text
        :return: end of sentence in document text
        """
        return self._end


def get_shape_category(token: str) -> str:
    """
    Get the shape category of a token
    :param token: The Token to get the shape of
    :return: The shape category
    """
    if re.match('^[\n]+$', token):  # IS LINE BREAK
        return 'NL'
    if any(char.isdigit() for char in token) and re.match('^[0-9.,]+$', token):  # IS NUMBER (E.G., 2, 2.000)
        return 'NUMBER'
    if re.fullmatch('[^A-Za-z0-9\t\n ]+', token):  # IS SPECIAL CHARS (E.G., $, #, ., *)
        return 'SPECIAL'
    if re.fullmatch('^[A-Z\-.]+$', token):  # IS UPPERCASE (E.G., AGREEMENT, INC.)
        return 'ALL-CAPS'
    if re.fullmatch('^[A-Z][a-z\-.]+$', token):  # FIRST LETTER UPPERCASE (E.G. This, Agreement)
        return '1ST-CAP'
    if re.fullmatch('^[a-z\-.]+$', token):  # IS LOWERCASE (E.G., may, third-party)
        return 'LOWER'
    if not token.isupper() and not token.islower():  # WEIRD CASE (E.G., 3RD, E2, iPhone)
        return 'MISC'
    return 'MISC'
