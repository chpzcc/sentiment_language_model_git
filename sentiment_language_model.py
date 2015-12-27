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
from utils import is_param_name, trunc_list, pad_list, save_checkpoint


class SentimentLanguageModel(object):

	@property
	def logger(self):
		return self._logger;

	@property
	def last_states(self):
		return self._last_states

	@property
	def output(self):
		return self._outputs

	def is_input(name):
		return name.endswith("initial_state") or name.endswith("data")


	def __init__(self, config, load_from_arg_dic=None):
		"""
			initialize some parameters
		"""
		if load_from_arg_dic==None:
			load_from_arg_dic = {}
		self.batch_size = batch_size = config.batch_size
		self.seq_len = seq_len = config.seq_len
		self._max_grad_norm = config.max_grad_norm
		size = config.hidden_size
		ctx = config.ctx
		vocab_size = config.vocab_size
		num_label = config.num_label
		self._logger = logger = Ylogger.Ylogger("test", "log/text.txt")

		"""
			build graph
		"""
		lstm_cell = []
		for l in range(config.num_layers):
			if l == 0:
				p = 0
			else:
				p = config.dropout
			lstm_cell.append(rnn_cell.LSTMCell(size, layeridx=l, dropout=p))
		senti_cell = rnn_cell.MultiRNNCell(lstm_cell)

		lstm_cell_extra = []
		for l in range(config.num_layers):
			if l == 0:
				p = 0
			else:
				p = config.dropout
			lstm_cell_extra.append(rnn_cell.LSTMCellWithExtraInput( \
												size, layeridx=l, dropout=p))
		lm_cell = rnn_cell.MultiRNNCell(lstm_cell_extra)

		inputs = []
		senti_embed_weight = mx.sym.Variable("senti_embed_weight")
		senti_cls_weight = mx.sym.Variable("senti_cls_weight")
		senti_cls_bias = mx.sym.Variable("senti_cls_bias")
		lm_embed_weight = mx.sym.Variable("lm_embed_weight")
		lm_cls_weight = mx.sym.Variable("lm_cls_weight")
		lm_cls_bias = mx.sym.Variable("lm_cls_bias")
		data = [mx.sym.Variable("t%d_data" % t) \
								for t in xrange(config.seq_len)]
		senti = [mx.sym.Variable("t%d_senti" % t) \
								for t in xrange(config.seq_len)]
		senti_mask = mx.sym.Variable("senti_mask")
		lm_mask = mx.sym.Variable("lm_mask")
		senti_params = [senti_embed_weight]
		lm_params = [lm_embed_weight]

		for l in xrange(config.num_layers):
			dic = {}
			dic["i2h_weight"] = mx.sym.Variable("senti_l%d_i2h_weight" % l)
			dic["i2h_bias"] = mx.sym.Variable("senti_l%d_i2h_bias" % l)
			dic["h2h_weight"] = mx.sym.Variable("senti_l%d_h2h_weight" % l)
			dic["h2h_bias"] = mx.sym.Variable("senti_l%d_h2h_bias" % l)
			senti_params.append(dic)

		for l in xrange(config.num_layers):
			dic = {}
			dic["i2h_weight"] = mx.sym.Variable("lm_l%d_i2h_weight" % l)
			dic["i2h_bias"] = mx.sym.Variable("lm_l%d_i2h_bias" % l)
			dic["h2h_weight"] = mx.sym.Variable("lm_l%d_h2h_weight" % l)
			dic["h2h_bias"] = mx.sym.Variable("lm_l%d_h2h_bias" % l)
			dic["s2h_weight"] = mx.sym.Variable("lm_l%d_s2h_weight" % l)
			dic["s2h_bias"] = mx.sym.Variable("lm_l%d_s2h_bias" % l)
			lm_params.append(dic)

		senti_cell = rnn_cell.EmbeddingWrapper(senti_cell, size)
		lm_cell = rnn_cell.EmbeddingWrapperWithExtraInput(lm_cell, size)

		initial_state = \
				[(mx.sym.Variable("%d_init_c" % i),
				  mx.sym.Variable("%d_init_h" % i))
				  for i in xrange(config.num_layers)]
		senti_outputs, senti_states = rnn.rnn(senti_cell, senti,
								  {"cell": senti_params,
								   "initial_state": initial_state})
		lm_outputs, lm_states = rnn.rnn(lm_cell,
										map(list, zip(data, senti_outputs)),
								  {"cell": lm_params,
								   "initial_state": initial_state})
		
		senti_ct = mx.sym.Concat(*senti_outputs, dim=0)
		lm_ct = mx.sym.Concat(*lm_outputs, dim=0)
		if config.dropout > 0.:
			senti_ct = mx.sym.Dropout(data=senti_ct, p=config.dropout)
			lm_ct = mx.sym.Dropout(data=lm_ct, p=config.dropout)
		senti_fc = mx.sym.FullyConnected(data=senti_ct,
								  		weight=senti_cls_weight,
								   		bias=senti_cls_bias,
								   		num_hidden=num_label)
		lm_fc = mx.sym.FullyConnected(data=lm_ct,
								  	  weight=lm_cls_weight,
								   	  bias=lm_cls_bias,
								   	  num_hidden=vocab_size)
		senti_label = mx.sym.Variable("senti_label")
		senti_sm = mx.sym.SoftmaxMaskOutput(data=senti_fc,
											label=senti_label,
											mask=senti_mask,
											name="senti_sm")
		lm_label = mx.sym.Variable("lm_label")
		lm_sm = mx.sym.SoftmaxMaskOutput(data=lm_fc,
										 label=lm_label,
										 mask=lm_mask,
										 name="lm_sm")
		senti_unpack_c = []
		senti_unpack_h = []
		for i, (c, h) in enumerate(senti_states[-1]):
			senti_unpack_c.append(mx.sym.BlockGrad(c, \
										name="senti_l%d_last_c" % i))
			senti_unpack_h.append(mx.sym.BlockGrad(h, \
										name="senti_l%d_last_h" % i))

		lm_unpack_c = []
		lm_unpack_h = []
		for i, (c, h) in enumerate(lm_states[-1]):
			lm_unpack_c.append(mx.sym.BlockGrad(c, \
										name="lm_l%d_last_c" % i))
			lm_unpack_h.append(mx.sym.BlockGrad(h, \
										name="lm_l%d_last_h" % i))
		rnn_sym = mx.sym.Group([senti_sm] + [lm_sm] +
								senti_unpack_c + senti_unpack_h +
								lm_unpack_c + lm_unpack_h)
		#dot = visualization.plot_network(rnn_sym)
		#dot.render("ptb.gv", view=True)

		"""
			produce interfaces for outside
		"""
		logger(rnn_sym.list_arguments())
		logger(rnn_sym.list_outputs())
		arg_names = rnn_sym.list_arguments()
		input_shapes = {}
		for name in arg_names:
			if name.endswith("init_c") or name.endswith("init_h"):
				input_shapes[name] = (batch_size, size)
			elif name.endswith("data"):
				input_shapes[name] = (batch_size, vocab_size)
			elif name.endswith("senti"):
				input_shapes[name] = (batch_size, 2)
			else:
				pass
		self._rnn_exec = rnn_exec = rnn_sym.simple_bind(ctx, "add", **input_shapes)

		arg_dict = dict(zip(arg_names, rnn_exec.arg_arrays))
		self.arg_dict = arg_dict
		self._param_blocks = []
		initializer = mx.initializer.Uniform(config.init_scale)
		for i, name in enumerate(arg_names):
			if is_param_name(name):
				#logger("init "+name)
				initializer(name, arg_dict[name])
				self._param_blocks.append((arg_dict[name],\
										   rnn_exec.grad_arrays[i]))
		for i, name in enumerate(arg_names):
			if name in load_from_arg_dic:
				logger("load parameter" + name)
				load_from_arg_dic[name].copyto(arg_dict[name])

		self._seq_data = [arg_dict["t%d_data" % t] for t in xrange(seq_len)]
		self._seq_senti = [arg_dict["t%d_senti" % t] for t in xrange(seq_len)]
		self._senti_mask = arg_dict["senti_mask"]
		self._lm_mask = arg_dict["lm_mask"]
		self._init_states = [(arg_dict["%d_init_c" % l],
							  arg_dict["%d_init_h" % l]) \
							  for l in xrange(config.num_layers)]
		out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))
		self._senti_last_states = [(out_dict["senti_l%d_last_c_output" % l],
							  		out_dict["senti_l%d_last_h_output" % l]) \
							  		for l in xrange(config.num_layers)]
		self._lm_last_states = [(out_dict["lm_l%d_last_c_output" % l],
							  	 out_dict["lm_l%d_last_h_output" % l]) \
							  	 for l in xrange(config.num_layers)]
		self._senti_labels = arg_dict["senti_label"]
		self._lm_labels = arg_dict["lm_label"]
		self._senti_outputs = out_dict["senti_sm_output"]
		self._lm_outputs = out_dict["lm_sm_output"]

		self._opt = mx.optimizer.create("sgd", wd=0, momentum=0,
										learning_rate=config.learning_rate)
		self._last_loss = 1e10
		self._updater = mx.optimizer.get_updater(self._opt)
		self._decay_when = config.decay_when
		self._lr_decay = config.lr_decay


	def update_lr(self, valid_loss):
		if (self._last_loss - self._decay_when < valid_loss):
			self._opt.lr *= self._lr_decay
			self.logger("lr changed"+str(self._opt.lr))
		self._last_loss = valid_loss

	def reset_states(self):
		for (c, h) in self._init_states:
			c[:] = 0.
			h[:] = 0.

	def save(self, prefix, epoch):
		arg_params = {}
		for name, tensor in self.arg_dict.items():
			if is_param_name(name):
				arg_params[name] = tensor
		save_checkpoint(prefix, epoch, arg_params)

	def set_inputs(self, input_word, sentiment_label, next_word):
		# input_word = (batch_size, seq_len)
		batch_size = self.batch_size
		seq_len = self.seq_len

		# truncate sentences to make it shorter
		input_word = [trunc_list(input_word[i], seq_len) \
						for i in xrange(len(input_word))]
		next_word = [trunc_list(next_word[i], seq_len) \
						for i in xrange(len(next_word))]

		self._sentence_lengths = sentence_lengths = map(len, input_word)

		# pad sentences
		padded_input = np.array([pad_list(input_word[i], self.seq_len) \
						for i in xrange(len(input_word))])
		padded_next_word = np.array([pad_list(next_word[i], self.seq_len) \
						for i in xrange(len(next_word))])
		label = np.array(sentiment_label)

		# bind input
		for seqidx in xrange(self.seq_len):
			x = padded_input[:, seqidx]

			# fixed sentiment
			mx.nd.onehot_encode(mx.nd.array(sentiment_label,
						ctx=self._seq_data[seqidx].context),
						out=self._seq_senti[seqidx])
			'''
			mx.nd.onehot_encode( \
						mx.nd.array(x, ctx=self._seq_data[seqidx].context),
						out=self._seq_senti[seqidx])
			'''
			mx.nd.onehot_encode( \
						mx.nd.array(x, ctx=self._seq_data[seqidx].context),
						out=self._seq_data[seqidx])

		for i in xrange(batch_size):
			self._senti_labels[i*seq_len : (i+1)*seq_len] = 0
			self._senti_mask[i*seq_len : (i+1)*seq_len] = 0
			self._lm_labels[i*seq_len : (i+1)*seq_len] = 0
			self._lm_mask[i*seq_len : (i+1)*seq_len] = 0

		# bind sentiment label
		for i in xrange(batch_size):
			pos_eos = (sentence_lengths[i]-1)*batch_size + i
			self._senti_labels[pos_eos : pos_eos+1] = label[i]
			self._senti_mask[pos_eos : pos_eos+1] = 1

		# bind language model label
		for i in xrange(batch_size):
			for j in xrange(sentence_lengths[i]-1):
				pos = (j+1)*batch_size + i
				self._lm_labels[pos : pos+1] = padded_next_word[i][j]
				self._lm_mask[pos : pos+1] = 1

	def Forward(self, is_train):
		self._rnn_exec.forward(is_train)
		seq_senti_label_probs = mx.nd.choose_element_0index( \
									self._senti_outputs,
									self._senti_labels)
		seq_lm_label_probs = mx.nd.choose_element_0index( \
									self._lm_outputs,
									self._lm_labels)

		batch_size = self.batch_size
		seq_len = self.seq_len

		# ingore all sentiment labels except the last one
		for i in xrange(batch_size):
			for j in xrange(seq_len):
				if j != self._sentence_lengths[i] - 1:
					pos = j*batch_size + i
					seq_senti_label_probs[pos : pos+1] = 1

		lm_nll = -np.sum(np.log(seq_lm_label_probs.asnumpy())) / batch_size
		senti_nll = -np.sum(np.log(seq_senti_label_probs.asnumpy())) / batch_size

		# calculate accuracy for sentiment task
		hit = 0
		np_senti_outputs = self._senti_outputs.asnumpy()
		np_senti_labels = self._senti_labels.asnumpy()
		np_senti_mask = self._senti_mask.asnumpy()
		for i in xrange(len(np_senti_outputs)):
			if np_senti_mask[i] == 0:
				continue
			hit += 1
			for j in xrange(len(np_senti_outputs[0])):
				if (j != np_senti_labels[i] and
						np_senti_outputs[i, j] >=
				  		np_senti_outputs[i, np_senti_labels[i]]):
					hit -= 1
					break

		return lm_nll, senti_nll, hit

	def Backward(self):
		self._rnn_exec.backward()
		norm = 0.
		for weight, grad in self._param_blocks:
			grad /= self.batch_size

			l2_norm = mx.nd.norm(grad).asscalar()
			norm += l2_norm*l2_norm
		norm = math.sqrt(norm)
		for idx, (weight, grad) in enumerate(self._param_blocks):
			if norm > self._max_grad_norm:
				grad *= (self._max_grad_norm / norm)

			self._updater(idx, grad, weight)
			grad[:] = 0.0