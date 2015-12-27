import sys
import mxnet as mx
import Ylogger
import reader
import rnn_cell
import rnn
import time
import numpy as np
import math
import random
from random import Random
from sentiment_language_model import SentimentLanguageModel


class SmallConfig(object):

	init_scale = 0.1
	learning_rate = 1.0#0.5
	max_grad_norm = 5
	num_layers = 1
	seq_len = 30#10
	hidden_size = 200#10
	max_epoch = 30
	dropout = 0.5#0
	lr_decay = 1
	decay_when = 0.0
	batch_size = 128#1
	vocab_size = 20000#100
	num_label = 2
	ctx = mx.gpu()
	"""
	init_scale = 0.1
	learning_rate = 0.05
	max_grad_norm = 5
	num_layers = 1
	seq_len = 10
	hidden_size = 10
	max_epoch = 500
	dropout = 0
	lr_decay = 0.9
	decay_when = 0.0
	batch_size = 2
	vocab_size = 5
	num_label = 2
	ctx = mx.gpu()
	"""


def run_epoch(m, data, is_train, verbose=False):
	epoch_size = len(data)
	start_time = time.time()
	lm_loss = 0.0
	senti_loss = 0.0
	acc = 0.0
	hit = 0
	iter_words = 0
	m.reset_states
	for step, (input_word, sentiment_label, next_word) \
				in enumerate(reader.stb_iterator(data, m.batch_size)):
		m.set_inputs(input_word, sentiment_label, next_word)
		cur_lm_loss, cur_senti_loss, cur_hit = m.Forward(is_train)
		lm_loss += cur_lm_loss
		senti_loss += cur_senti_loss
		hit += cur_hit
		iter_words += m.seq_len
		acc = hit / (step+1) / float(m.batch_size)

		if is_train == True:
			m.Backward()

		# print info every 100 batches
		if verbose and step % 100 == 0:
			m.logger("[%.3f] lm_loss: %.3f senti_loss: %.3f (acc: %.3f) speed: %.0f wps" %
					 (step*m.batch_size / float(epoch_size),
					 lm_loss / (step+1),
					 senti_loss / (step+1),
					 acc,
					 iter_words*m.batch_size / (time.time()-start_time)))

	return lm_loss, senti_loss, acc


def get_config():
	return SmallConfig()


def main(load_from_epoch=None):
	prefix = 'stb'
	raw_data = reader.stb_raw_data("./data/")
	train_data, valid_data, test_data, _ = raw_data

	config = get_config()
	if load_from_epoch == None:
		m = SentimentLanguageModel(config=config)
	else:
		arg_params = load_checkpoint(prefix, load_from_epoch)
		m = SentimentLanguageModel(config, arg_params)
		m.logger("load from %s %d" % (prefix, load_from_epoch))

	last_loss = 1e10
	for i in xrange(config.max_epoch):
		my_random = Random()
		my_random.seed(1)
		my_random.shuffle(train_data)

		train_lm_loss, train_senti_loss, train_acc = run_epoch(m, train_data, True, True)
		m.logger("Epoch: %d Training LM Loss: %.3f Training Senti Loss: %.3f (Accuracy: %.3f)" %
								(i + 1, train_lm_loss, train_senti_loss, train_acc))

		valid_lm_loss, valid_senti_loss, valid_acc = run_epoch(m, valid_data, False, False)
		m.logger("Epoch: %d Valid LM Loss: %.3f Valid Senti Loss: %.3f (Accuracy: %.3f)" %
								(i + 1, valid_lm_loss, valid_senti_loss, valid_acc))

		m.update_lr(valid_lm_loss)
		m.save(prefix, i)

if __name__ == "__main__":
	main()