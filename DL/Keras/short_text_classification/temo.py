import cjdpy
import random

"""
vocabulary size
	emotion_analysis: 5000
	news_title: 5000
	sentiment_classification: 15000
	topic_classification:30000
"""
def train_test_data():
	data = cjdpy.load_csv('data/topic_classification/text.con.txt')
	train_file = 'data/topic_classification/train.txt'
	test_file = 'data/topic_classification/test.txt'
	random.shuffle(data)
	line=10000
	cjdpy.save_csv(data[:line],test_file)
	cjdpy.save_csv(data[line:], train_file)

def get_activations(model, inputs, print_shape_only=False, layer_name=None):

	activations = []
	inp = model.input
	if layer_name is None:
		outputs = [layer.output for layer in model.layers]
	else:
		outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs

	funcs = [K.function([inp], [out]) for out in outputs]  # evaluation functions
	# print('funcs: ', funcs)
	res = funcs[0](inputs)
	print(res)
	print(type(res))
	print(res[0].shape)
	for i in range(100):
		print(res[0][i])
	assert False
	layer_outputs = [func(inputs)[0] for func in funcs]
	for layer_activations in layer_outputs:
		activations.append(layer_activations)
		if print_shape_only:
			print(layer_activations.shape)
		else:
			print(layer_activations)
	# print('activation: ', activations)
	return activations


from keras.layers import *
from keras.models import Model

input1 = Input((3,))
input2 = Input((3,))
output = Add(name='add')([input1, input2])
m = Model([input1,input2], output)
m.summary()

inp = m.input
print(inp)
for layer in m.layers:
	if layer.name == 'add':
		out = layer.output
func = K.Function(inp, [out])
X = [[1,2,3], [6,5,4]]
X = [[1,2,3], [6,5,4]]
res = func([X,X])
print(res)
