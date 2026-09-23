"""Rebuild the HumBug MozzBNN mosquito event detector (Keras 2.3 .hdf5) as ONNX.

The Keras file stores its MC-dropout layers as Python 3.7 lambda bytecode, which modern
TensorFlow cannot load, so the graph is rebuilt here from the stored weights:
  input [N,1,30,128] (channels-first log-mel window, 1.92 s at 8 kHz)
  Conv3x3(32)+ReLU, MaxPool2, Dropout, Conv3x3(64)+ReLU, MaxPool2, Dropout,
  Conv3x3(64)+ReLU, Dropout, Conv3x3(64)+ReLU, Dropout,
  Flatten (Keras channels_first Flatten = NHWC order), Dense 128+ReLU, Dropout, Dense 2+Softmax
Dropout is kept active (training_mode=True) exactly like K.dropout in the original lambdas:
feed the same window several times in one batch to get Monte Carlo samples.

Usage: python export_humbug.py neurips_2021_humbugdb_keras_bnn_best.hdf5 dist/model
"""
import hashlib, json, os, sys
import h5py, numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper

src = sys.argv[1] if len(sys.argv) > 1 else 'neurips_2021_humbugdb_keras_bnn_best.hdf5'
root = sys.argv[2] if len(sys.argv) > 2 else 'dist/model'
w = h5py.File(src, 'r')['model_weights']
get = lambda layer, name: np.array(w[layer][layer][name + ':0'], dtype=np.float32)

inits, nodes = [], []
def const(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name=name)); return name
ratio = const('dropout_ratio', np.array(0.2, dtype=np.float32))
train = const('dropout_training', np.array(True))

x = 'mel'
def conv(layer, x):
    k = get(layer, 'kernel').transpose(3, 2, 0, 1)  # (kh,kw,in,out) -> (out,in,kh,kw)
    nodes.append(helper.make_node('Conv', [x, const(layer + '_W', k), const(layer + '_B', get(layer, 'bias'))], [layer], kernel_shape=[3, 3]))
    nodes.append(helper.make_node('Relu', [layer], [layer + '_relu'])); return layer + '_relu'
def pool(x, name):
    nodes.append(helper.make_node('MaxPool', [x], [name], kernel_shape=[2, 2], strides=[2, 2])); return name
def drop(x, name):
    nodes.append(helper.make_node('Dropout', [x, ratio, train], [name])); return name
def dense(layer, x, act):
    nodes.append(helper.make_node('Gemm', [x, const(layer + '_W', get(layer, 'kernel')), const(layer + '_B', get(layer, 'bias'))], [layer]))
    nodes.append(helper.make_node(act, [layer], [layer + '_' + act.lower()], **({'axis': 1} if act == 'Softmax' else {}))); return layer + '_' + act.lower()

x = drop(pool(conv('conv2d_9', x), 'pool1'), 'drop1')
x = drop(pool(conv('conv2d_10', x), 'pool2'), 'drop2')
x = drop(conv('conv2d_11', x), 'drop3')
x = drop(conv('conv2d_12', x), 'drop4')
nodes.append(helper.make_node('Transpose', [x], ['nhwc'], perm=[0, 2, 3, 1]))  # Keras Flatten(channels_first) flattens as NHWC
nodes.append(helper.make_node('Flatten', ['nhwc'], ['flat'], axis=1))
x = drop(dense('dense_5', 'flat', 'Relu'), 'drop5')
out = dense('dense_6', x, 'Softmax')
nodes.append(helper.make_node('Identity', [out], ['prob']))

graph = helper.make_graph(nodes, 'humbug_mozzbnn_med',
    [helper.make_tensor_value_info('mel', TensorProto.FLOAT, ['N', 1, 30, 128])],
    [helper.make_tensor_value_info('prob', TensorProto.FLOAT, ['N', 2])], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], producer_name='mosquito-scanner')
model.ir_version = 8
onnx.checker.check_model(model)
b = model.SerializeToString()
os.makedirs(root, exist_ok=True)
open(os.path.join(root, 'humbug-med.onnx'), 'wb').write(b)
json.dump({'file': 'humbug-med.onnx', 'size': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
           'input': [None, 1, 30, 128], 'output': [None, 2], 'classes': ['background', 'mosquito'],
           'sr': 8000, 'n_fft': 2048, 'hop': 512, 'n_mels': 128, 'win': 30, 'mc_samples': 10,
           'source': 'https://github.com/HumBug-Mosquito/MozzBNN', 'weights': 'neurips_2021_humbugdb_keras_bnn_best.hdf5',
           'weights_sha256': hashlib.sha256(open(src, 'rb').read()).hexdigest(), 'license': 'MIT'},
          open(os.path.join(root, 'humbug-manifest.json'), 'w'), indent=1)
print('ONNX bytes', len(b))
