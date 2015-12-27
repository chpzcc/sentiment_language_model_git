import mxnet as mx
import copy

def is_param_name(name):
		return name.endswith("weight") or name.endswith("bias") or\
		name.endswith("gamma") or name.endswith("beta")


def save_checkpoint(prefix, epoch, arg_params):
	save_dict = {('arg:%s '% k) : v for k,v in arg_params.items()}
	param_name = '%s-%04d.params' % (prefix, epoch)
	mx.nd.save(param_name, save_dict)
	print 'Saved checkpoint to \"%s\"' % param_name


def load_checkpoint(prefix, epoch):
	save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
	arg_params = {}
	for k, v in save_dict.items():
		tp, name = k.split(':', 1)
		arg_params[name.strip()] = v
	return arg_params

def trunc_list(x, seq_len):
	trunc_x = copy.copy(x)
	if (len(trunc_x) > seq_len):
		trunc_x = trunc_x[0 : seq_len]
	return trunc_x

def pad_list(x, seq_len):
	pad_x = copy.copy(x)
	for i in xrange(len(x), seq_len):
		pad_x.append(0)
	return pad_x

def shift_list(x):
	shift_x = x[1:]
	shift_x.append(0)
	return shift_x