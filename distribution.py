#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Kamel ABDELOUAHAB
# DREAM - Institut Pascal - Universite Clermont Auvergne
# Last update : 23-07-2018
# distribution.py : Plots the ditribution of CNN activations, weights and compares models

import os
import sys
import io
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'grid.color' : 'k',
    'grid.linestyle': 'dashdot',
    'grid.linewidth': 0.6,
    'font.family': 'Linux Biolinum O',
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.facecolor' : 'white'
   }
rcParams.update(params)


def quantizeArray(float_weight,bitwidth=8):
    scale_factor = 2**(bitwidth-1) - 1
    scaled_data = np.round(float_weight * scale_factor)
    return np.array(scaled_data, dtype=int)

def listToNumpy(in_list):
    np_data = np.empty([len(in_list), m],dtype=int)
    for i in range(len(in_list)):
        np_data[i,0:len(in_list[i])] = in_list[i]
    return np_data


CAFFE_ROOT = os.environ['CAFFE_ROOT']
CAFFE_PYTHON_LIB = CAFFE_ROOT+'/python'
sys.path.insert(0, CAFFE_PYTHON_LIB)
os.environ['GLOG_minloglevel'] = '2'                         # Supresses Display on console
import caffe;

# Replace with Path to proto and modelfile
proto_file = "/home/kamel/Seafile/CNN-Models/alexnet.prototxt"
model_file = "/home/kamel/Seafile/CNN-Models/alexnet.caffemodel"
image = caffe.io.load_image("./cat.jpg")
hist1_pdf = "./act_hist.pdf"
hist2_pdf = "./param_hist.pdf"


net = caffe.Net(proto_file,1,weights=model_file)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
#transformer.set_raw_scale('data', 255)                    # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))               # swap channels from RGB to BGR
net.blobs['data'].reshape(50, 3, 227, 227)
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image
net.forward()

conv_layer_name = []
conv_layer_data = []
conv_layer_param = []

fc_layer_name = []
fc_layer_data = []
fc_layer_param = []

data = net.blobs['data'].data.ravel()
conv1 = net.blobs['conv1'].data.ravel()
conv5 = net.blobs['conv4'].data.ravel()

#for l in net._layer_names:
    #layer_id = list(net._layer_names).index(l)
    #layer_type =  net.layers[layer_id].type
    #if (layer_type == 'Convolution'):
        #conv_layer_name.append(l)
        #this_layer_data = net.blobs[l].data.ravel()
        #this_layer_param = net.params[l][0].data.ravel()
        #conv_layer_data.append(this_layer_data)
        #conv_layer_param.append(this_layer_param)
    #if (layer_type == 'InnerProduct'):
        #fc_layer_name.append(l)
        #this_layer_data = net.blobs[l].data.ravel()
        #this_layer_param = net.params[l][0].data.ravel()
        #fc_layer_data.append(this_layer_data)
        #fc_layer_param.append(this_layer_param)

#m = np.max(len(conv_layer_data[0]))
#for i in range(len(conv_layer_data)):
    #if len(conv_layer_data[i]) > m:
        #m = np.max(len(conv_layer_data[i]))

#np_conv_data = np.empty([len(conv_layer_data), m],dtype=int)
#for i in range(len(conv_layer_data)):
    #np_conv_data[i,0:len(conv_layer_data[i])] = conv_layer_data[i]
#np_conv_data = quantizeArray(np_conv_data,8)
#XBASE, YBASE = 2, 10                                        # Log base used

# # Histogram 1: Conv data layer per layer                       # Range of values
#bins = 2 ** np.linspace(-2,              # Min value
                        #12,              # Max value
                        #14)              # number of bins
                        
logbins = np.logspace(-5, 6, 12, base=2)


plt.figure(figsize=(15,3))
plt.hist(conv5,
        bins=logbins,
        density=True,
        color='red',
        #stacked=True,
        alpha=0.8)    
plt.gca().set_xscale('log', basex=2)
plt.legend(loc='upper right')
plt.grid(linestyle='dotted')
plt.show()

plt.figure(figsize=(15,3))
plt.hist(data,
        bins=logbins,
        density=True,
        color='blue',
        #stacked=True,
        alpha=0.8)      
plt.gca().set_xscale('log', basex=2)
plt.legend(loc='upper right')
plt.grid(linestyle='dotted')
plt.show()
#plt.savefig(hist1_pdf, bbox_inches ='tight')
## Histogram 2: conv data vs conv param
#MIN_VALUE, MAX_VALUE = -2.0**8, 2.0**8                  # Range of values
#all_conv_data =  np.concatenate(np_conv_data).ravel()
#all_conv_param = np.concatenate(conv_layer_param).ravel()
#all_conv_param = quantizeArray(all_conv_param,8)
#power_array_pos = np.linspace(0,12,13)
#power_array_neg = np.linspace(1,7,7)
#bins = np.append(-2 ** power_array_neg[::-1], 2 ** power_array_pos)
#plt.figure(figsize=(15,3))
#plt.hist([all_conv_data, all_conv_param],
        #bins=bins,
        #label=['Activations', 'Weights'],
        #alpha=0.8)
#plt.gca().set_xscale('symlog', basex=XBASE)
#plt.gca().set_yscale('log', basey=YBASE)
#plt.legend(loc='upper right')
#plt.grid(linestyle='dotted')
#plt.show()
#plt.savefig(hist2_pdf, bbox_inches ='tight')


## Histogram 3: weights Alexnet vs weights Alexnet_compressed
#alexnet_layer_param = []
#alexnet_compressed_layer_param = []
#proto_file = "/home/kamel/Seafile/CNN-Models/alexnet.prototxt"
#model_file = "/home/kamel/Seafile/CNN-Models/alexnet.caffemodel"
#alexnet = caffe.Net(proto_file,1,weights=model_file)
#for l in alexnet._layer_names:
    #layer_id = list(alexnet._layer_names).index(l)
    #layer_type =  alexnet.layers[layer_id].type
    #if (layer_type == 'Convolution'):
        #this_layer_param = alexnet.params[l][0].data.ravel()
        #alexnet_layer_param.append(this_layer_param)

#proto_file = "/home/kamel/Seafile/CNN-Models/alexnet_compressed.prototxt"
#model_file = "/home/kamel/Seafile/CNN-Models/alexnet_compressed.caffemodel"
#alexnet_compressed = caffe.Net(proto_file,1,weights=model_file)
#for l in alexnet_compressed._layer_names:
    #layer_id = list(alexnet_compressed._layer_names).index(l)
    #layer_type =  alexnet_compressed.layers[layer_id].type
    #if (layer_type == 'Convolution'):
        #this_layer_param = alexnet_compressed.params[l][0].data.ravel()
        #alexnet_compressed_layer_param.append(this_layer_param)

 #alexnet_layer_param = listToNumpy(alexnet_layer_param)
 #alexnet_compressed_layer_param = listToNumpy(alexnet_compressed_layer_param)
#alexnet_original = np.concatenate(alexnet_layer_param).ravel()
#alexnet_compressed = np.concatenate(alexnet_compressed_layer_param).ravel()
#print(alexnet_original.shape)
#print(alexnet_compressed.shape)

#alexnet_original = quantizeArray(alexnet_original,9)
#alexnet_compressed = quantizeArray(alexnet_compressed,9)

#MIN_VALUE, MAX_VALUE = -2.0**6, 2.0**6
#power_array_pos = np.linspace(0,6,7)
#power_array_neg = np.linspace(1,7,7)
#bins = np.append(-2 ** power_array_neg[::-1], 2 ** power_array_pos)
#plt.figure(figsize=(12,3))
#plt.hist([alexnet_original, alexnet_compressed],
         #bins=bins,
         #label=['Alexnet', 'Alexnet Compressed'],
         #alpha=0.8)
#plt.gca().set_xscale('symlog', basex=XBASE)
#plt.gca().set_yscale('log', basey=YBASE)
#plt.legend(loc='upper right')
#plt.grid(linestyle='dotted')
#plt.show()
