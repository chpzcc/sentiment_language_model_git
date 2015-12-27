import sys
sys.path.insert(0, "../../python")
import mxnet as mx


from collections import namedtuple

class RNNCell(object):
	def __call__(self, inputs, state, param=None):
		raise NotImplementedError("Abstract method")
	@property
	def param(self):
		raise NotImplementedError("Abstract method")

class LSTMCell(RNNCell):

	def __init__(self, num_hidden,layeridx,dropout = 0., forget_bias = 1.0):
		self._forget_bias = forget_bias
		self._num_hidden = num_hidden
		self._dropout = dropout
		self._layeridx =layeridx

	@property
	def layeridx(self):
		return self._layeridx


	def __call__(self, inputs, state,param):
		if self._dropout>0.:
			inputs = mx.sym.Dropout(data=inputs, p=self._dropout)
		(c,h) =state
		i2h = mx.sym.FullyConnected(data=inputs,
	    				weight=param["i2h_weight"],
	    				bias = param["i2h_bias"],
	                                num_hidden=self._num_hidden * 4)
		h2h = mx.sym.FullyConnected(data=h,
	    			weight=param["h2h_weight"],
	    				bias = param["h2h_bias"],
	                                num_hidden=self._num_hidden * 4)			    
	
		gates = i2h + h2h
	   	slice_gates = mx.sym.SliceChannel(gates, num_outputs=4)
	   	in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
	   	in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
	   	forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
	   	out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
	   	next_c = (forget_gate * c) + (in_gate * in_transform)
	   	next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
	   	return next_h, (next_c,next_h)


class LSTMCellWithExtraInput(RNNCell):

	def __init__(self, num_hidden,layeridx,dropout = 0., forget_bias = 1.0):
		self._forget_bias = forget_bias
		self._num_hidden = num_hidden
		self._dropout = dropout
		self._layeridx =layeridx

	@property
	def layeridx(self):
		return self._layeridx


	def __call__(self, inputs, state,param):
		"""
			main input: input[0]
			extra input: input[1]
		"""

		if self._dropout>0.:
			inputs[0] = mx.sym.Dropout(data=inputs[0], p=self._dropout)
			inputs[1] = mx.sym.Dropout(data=inputs[1], p=self._dropout)

		(c,h) =state
		i2h = mx.sym.FullyConnected(data=inputs[0],
	    				weight=param["i2h_weight"],
	    				bias = param["i2h_bias"],
	                                num_hidden=self._num_hidden * 4)
		# projection for extra input
		s2h = mx.sym.FullyConnected(data=inputs[1],
	    				weight=param["s2h_weight"],
	    				bias = param["s2h_bias"],
	    							num_hidden=self._num_hidden * 4)
		h2h = mx.sym.FullyConnected(data=h,
	    				weight=param["h2h_weight"],
	    				bias = param["h2h_bias"],
	                                num_hidden=self._num_hidden * 4)
	
		gates = i2h + s2h + h2h
	   	slice_gates = mx.sym.SliceChannel(gates, num_outputs=4)
	   	in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
	   	in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
	   	forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
	   	out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
	   	next_c = (forget_gate * c) + (in_gate * in_transform)
	   	next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
	   	return next_h, (next_c,next_h)


class EmbeddingWrapper(RNNCell):
	def __init__(self, cell, embedding_classes):
		if not isinstance(cell, MultiRNNCell):
			raise TypeError("cell must be an instance of MultiRNNCell")
		assert embedding_classes > 0
		self._cell = cell
		self._embedding_classes = embedding_classes		
	@property
	def num_lstm_layer(self):
		return self._cell.num_lstm_layer		
		
	def __call__(self,inputs,state,params):	
		assert len(params) == self.num_lstm_layer+1
		embed_weight = params[0]
		embedded = mx.sym.FullyConnected(data=inputs, weight=embed_weight,
                                      num_hidden = self._embedding_classes,
                                       no_bias=True)
		#call MultiRNNCell
		params = [params[i+1] for i in range(len(params)-1)]
		return self._cell(embedded,state,params)


class EmbeddingWrapperWithExtraInput(RNNCell):
	def __init__(self, cell, embedding_classes):
		if not isinstance(cell, MultiRNNCell):
			raise TypeError("cell must be an instance of MultiRNNCell")
		assert embedding_classes > 0
		self._cell = cell
		self._embedding_classes = embedding_classes		
	@property
	def num_lstm_layer(self):
		return self._cell.num_lstm_layer		
		
	def __call__(self,inputs,state,params):	
		assert len(params) == self.num_lstm_layer+1
		embed_weight = params[0]
		embedded = mx.sym.FullyConnected(data=inputs[0], weight=embed_weight,
                                      num_hidden = self._embedding_classes,
                                       no_bias=True)
		#call MultiRNNCell
		params = [params[i+1] for i in range(len(params)-1)]
		return self._cell([embedded] + inputs[1:], state,params)


class MultiRNNCell(RNNCell):
	def __init__(self,cells):
		self._cells = cells
		
	@property
	def num_lstm_layer(self):
		return len(self._cells)


	def __call__(self, inputs, states,params):
		assert len(states) == self.num_lstm_layer
		next_states = []
		current = inputs
		for i,cell in enumerate(self._cells):
			current, next_state = cell(current, states[i],params[i])
			next_states.append(next_state)
			assert i == cell.layeridx
			
		return current, next_states


