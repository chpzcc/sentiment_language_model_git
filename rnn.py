import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import rnn_cell

def rnn(cell, inputs, params):
	"""
		params = {
			"initial_state" : list of (mx.sym.Variable("init_c"),mx.sym.Variable("init_h"))
			"cell" : list of params
					if cells have embedding, then the first element of params will be embed_weight
					The rest of params will be dict of lstm weight					
		}
	"""
	if (not isinstance(cell, rnn_cell.MultiRNNCell) and
			not isinstance(cell, rnn_cell.EmbeddingWrapper) and
			not isinstance(cell, rnn_cell.EmbeddingWrapperWithExtraInput)):
		raise TypeError("cell must be an instance of MultiRNNCell")
	if not isinstance(inputs, list):
		raise TypeError("inputs must be a list")
	if not inputs:
		raise ValueError("inputs must not be empty")
	outputs = []
	states = []
	state = params['initial_state']

	for t, input_ in enumerate(inputs):		
		output, state = cell(input_, state, params['cell'])
		outputs.append(output)
		states.append(state)
	return (outputs, states)
