import collections
import os
import sys
import time
from random import Random
from utils import shift_list

import numpy as np

import gfile


def _read_words(filename):
	with gfile.GFile(filename, "r") as f:
		# wait to be debugged
		return " ".join(f.readlines()[::2]).split()


def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id


def _split(x):
	return x.split()


def _read_sentences(filename):
	with gfile.GFile(filename, "r") as f:
		lines = f.readlines()
		return zip(map(_split, lines[::2]), lines[1::2])

def _transform(word_to_id, word):
	if word in word_to_id:
		return word_to_id[word]
	else:
		return 0


def _file_to_word_ids(filename, word_to_id):
	data = _read_sentences(filename)
	return [([_transform(word_to_id, word) for word in words], label) \
			for words, label in data]


def stb_raw_data(data_path = None):
	"""Load STB (stanford treebank) raw data from directory "data_path"
	"""

	train_path = os.path.join(data_path, "stb.train.txt")
	valid_path = os.path.join(data_path, "stb.valid.txt")
	test_path = os.path.join(data_path, "stb.test.txt")

	word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path, word_to_id)

	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	vocabulary = len(word_to_id)

	return train_data, valid_data, test_data, vocabulary


def stb_iterator(raw_data, batch_size):
	"""Iterate over the raw STB data

	This generate batch size pointers into the raw PTB data, and allows
	minibatch iteration along these pointers.

	Args:
	  raw_data: one of the raw data outputs from stb_raw_data.
	  batch_size: int, the batch size.

	Yield:
	  Pairs of the batched data, each three matrices of shape
	  [batch, max_sentence_len], [batch, 1] and [batch, max_sentence_len].
	  The second element of the tuple is the sentiment label for each
	  sentences.
	"""

	data_len = len(raw_data)
	batch_len = data_len//batch_size

	for i in xrange(batch_len):
		batch_data = raw_data[i*batch_size : (i+1)*batch_size]
		input_word = zip(*batch_data)[0]
		sentiment_label = map(int, zip(*batch_data)[1])
		next_word = map(shift_list, input_word)
		yield(input_word, sentiment_label, next_word)