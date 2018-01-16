#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import io
import itertools
import numpy as np
import matplotlib.pyplot as plt


CAFFE_ROOT = os.environ['CAFFE_ROOT']
CAFFE_PYTHON_LIB = CAFFE_ROOT+'/python'
sys.path.insert(0, CAFFE_PYTHON_LIB)
os.environ['GLOG_minloglevel'] = '2'         # Supresses Display on console
import caffe;


proto_file = "/home/kamel/dev/nets/alexnet_eval.prototxt"
model_file = "/home/kamel/dev/nets/alexnet.caffemodel"
net = caffe.Net(proto_file,1,weights=model_file)
net.forward()

conv_layer_name = []
conv_layer_data = []
conv_layer_param = []

fc_layer_name = []
fc_layer_data = []
fc_layer_param = []

for l in net._layer_names:
	layer_id = list(net._layer_names).index(l)
	layer_type =  net.layers[layer_id].type
	if (layer_type == 'Convolution'):
		conv_layer_name.append(l)
		this_layer_data = net.blobs[l].data.ravel()
		this_layer_param = net.params[l][0].data.ravel()
		conv_layer_data.append(this_layer_data)
		conv_layer_param.append(this_layer_param)
	if (layer_type == 'InnerProduct'):
		fc_layer_name.append(l)
		this_layer_data = net.blobs[l].data.ravel()
		this_layer_param = net.params[l][0].data.ravel()
		fc_layer_data.append(this_layer_data)
		fc_layer_param.append(this_layer_param)


## Histogram 1: Conv data layer per layer
XBASE, YBASE = 2, 10                                        # Log base used
MIN_VALUE, MAX_VALUE = 0, 2.0**10                         # Range of values
bins = 2 ** np.linspace(0,                                      # Min value
                        np.log2(MAX_VALUE),                     # Max value
						np.log2(MAX_VALUE)+1)               #number of bins
plt.hist(conv_layer_data,
         bins=bins,
		 label=conv_layer_name,
		 alpha=0.8)
plt.gca().set_xscale('log', basex=XBASE)
plt.gca().set_yscale('log', basey=YBASE)
plt.legend(loc='upper right')
plt.grid(linestyle='dotted')
plt.show()



## Histogram 2: conv data vs conv param
MIN_VALUE, MAX_VALUE = 2.0**-5, 2.0**10                  # Range of values
all_conv_data =  np.concatenate(conv_layer_data).ravel()
all_conv_param = np.concatenate(conv_layer_param).ravel()
power_array_pos = np.linspace(-3,10,14)
power_array_neg = np.linspace(-3,0,4)
bins = np.append(-2 ** power_array_neg[::-1], 2 ** power_array_pos)
plt.hist([all_conv_data, all_conv_param],
         bins=bins,
		 label=['Activations', 'Weights'],
		 alpha=0.8)
plt.gca().set_xscale('symlog', basex=XBASE)
plt.gca().set_yscale('log', basey=YBASE)
plt.legend(loc='upper right')
plt.grid(linestyle='dotted')
plt.show()
