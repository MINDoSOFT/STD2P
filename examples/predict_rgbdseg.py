# import required modules and models
import numpy as np
import time
import glob
import random
from PIL import Image
import cv2

caffe_root = '../caffe-std2p/'

import sys
sys.path.insert(0,caffe_root + 'python')

import caffe
import os
import argparse

import scipy.io as sio

data_path = '../../../data/for_std2p'
output_path = './output/rgbdseg'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Used to draw segmentation result
color_panel = np.array([[128,128,128], [255,0,0], [0,255,0], [0,0,255], [255,255,0],
               [255,0,255], [0,255,255], [64,255,0], [128,128,0], [128,0,128],
               [0,128,128], [64,0,192], [0,64,96], [192,64,0], [192,0,64],
               [0,192,64], [0,64,192], [128,0,255], [128,255,0], [255,128,0],
               [255,0,128], [0,128,255], [0,255,128], [32,32,192], [32,192,32],
               [192,32,32], [32,64,128], [32,128,64], [64,32,128], [64,100,16],
               [128,64,32], [128,32,64], [16,72,200], [72,16,200], [200,72,16],
               [72,200,16], [200,16,72], [72,150,100], [50,100,200],[60,96,96]])

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fcn_prototxt', default='./prototxt/fcn_16s_rgbd.prototxt')
parser.add_argument('-s', '--std2p_prototxt', default='./prototxt/aggregation_std2p.prototxt')
parser.add_argument('-g', '--gpu_id', default=0)
parser.add_argument('-m', '--our_model', default='./models/fcn-16s-rgbd-nyud2.caffemodel')
parser.add_argument('-i', '--img_order', default=1)

args = parser.parse_args()
gpu_id = args.gpu_id
fcn_prototxt = args.fcn_prototxt
std2p_prototxt = args.std2p_prototxt
our_model = args.our_model
img_order = int(args.img_order)

f = open('../config/my_target.txt', 'r')
targets = f.readlines()
f.close()
 
for index_, text_ in enumerate(targets):
  targets[index_] = int(text_) 
   
f = open('../config/my_order.txt', 'r')
orders = f.readlines()
f.close()
 
for index_, text_ in enumerate(orders):
  orders[index_] = int(text_) 

orders = np.array(orders).reshape(len(orders)/2,2)

f = open('../config/my_folder.txt', 'r')
folders = f.readlines()
f.close()

for index_, text_ in enumerate(folders):
  folders[index_] = text_[0:len(text_)-1]

random.seed(10)

# Segment an image using fully convolutional neural networks.
# Set Caffe to GPU mode, load the net in the test phase for inference, and configure input preprocessing.
caffe.set_mode_gpu()
caffe.set_device(int(gpu_id))

# select 11 views to predict target frame
nFrames = 11
max_segments = 550 # If not 550 it crashes at net_integrate.blobs['label_set'].data[...] = label_set

folder = folders[img_order-1]

target = targets[img_order-1]

order = orders[img_order-1,:]

frames = range(order[0],order[1]+1,3)
random.shuffle(frames)
frames = np.array(frames[0:nFrames])
if sum(frames == target) == 0:
    frames[0] = target

frames.sort()


net = caffe.Net(fcn_prototxt,
    our_model,
    caffe.TEST)

rgb_data = np.zeros((1,3,425,560))
hha_data = np.zeros((1,3,425,560))
post_multiple = np.zeros((nFrames,40,425,560)) # 40 class for nyudv2 dataset
corr_data = np.zeros((nFrames,1,425,560))
	
transformer = caffe.io.Transformer({'bgr_data': net.blobs['bgr_data'].data.shape, 'hha_data': net.blobs['hha_data'].data.shape})
transformer.set_transpose('bgr_data', (2,0,1))
transformer.set_mean('bgr_data', np.array([104.00698793,116.66876762,122.67891434])) # mean pixel
transformer.set_raw_scale('bgr_data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('bgr_data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
transformer.set_transpose('hha_data', (2,0,1))
transformer.set_mean('hha_data', np.array([118.477,94.076,132.431])) # mean pixel
transformer.set_raw_scale('hha_data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

# Classify the image by reshaping the net for the single input then doing the forward pass.
TimeStart = int(round(time.time() * 1000))

print frames
for iFrame in range(len(frames)):
    print folder
    print frames[iFrame]
    rgb_path = data_path + '/%s/%05d_color.png' % (folder, frames[iFrame])
    hha_path = data_path + '/%s/%05d_hha.png' % (folder, frames[iFrame])
    print rgb_path
    print hha_path

    img_rgb = caffe.io.load_image(rgb_path)
    img_hha = caffe.io.load_image(hha_path)
    if frames[iFrame] == target:
        print rgb_path

    # img_rgb = img_rgb[45:470,40:600,:]
    img_hha = img_hha[:,:,:]

    net.blobs['bgr_data'].data[...] = transformer.preprocess('bgr_data', img_rgb)
    net.blobs['hha_data'].data[...] = transformer.preprocess('hha_data', img_hha)
    net.forward()

    upscore = net.blobs['upscore'].data
    upscore = np.array(upscore)

    post_multiple[iFrame,:,:,:] = upscore[0]

    corr_path = data_path + '/%s/%04d/correspondences/%05d.txt' % (folder, img_order, frames[iFrame])
    print(corr_path)
    f = open(corr_path, 'r')
    corr = f.readlines()
    f.close()
	
    for index_, text_ in enumerate(corr):
        corr[index_] = int(text_)


    corr = np.array(corr).reshape(560,425).transpose()
    corr_data[iFrame,:,:,:] = corr

    if frames[iFrame] == target:

        print('Corr')
        print(corr.shape)

#        #Keep only the most occuring unique correspondences (to fit max_segments)
#        from collections import Counter
#        flatCorr = corr.flatten()
#        temp1 = [x for _,x in sorted(zip(Counter(flatCorr).values(),Counter(flatCorr).keys()), reverse=True)]
#        print('Frequent Temp1')
#        print(temp1)
#        #temp1 = np.resize(temp1, [max_segments])
#        temp1 = np.array(temp1[0:max_segments])

        temp1 = np.unique(corr)
        #print('unique(corr)')
        #print(temp1)
        #print(temp1.shape)

        temp1 = temp1.reshape([temp1.size])
        #temp1 = np.resize(temp1, [max_segments])

        #print('Temp1')
        #print(temp1.shape)
        number = temp1.size
        temp = -1*np.ones([max_segments])

        #print(corr_path)
        #print('Temp1')
        #print(temp1.shape)
        #print('Temp[0:temp1.size]')
        #print(temp[0:temp1.size].shape)
        #print('Temp')
        #print(temp.shape)

        temp[0:temp1.size] = temp1
        label_set = temp

        target_number = iFrame

del net

net_integrate = caffe.Net(std2p_prototxt,
    our_model,
    caffe.TEST)

net_integrate.blobs['data'].data[...] = post_multiple
net_integrate.blobs['correlation'].data[...] = corr_data
net_integrate.blobs['target'].data[...] = target_number
net_integrate.blobs['segment_number'].data[...] = number
net_integrate.blobs['label_set'].data[...] = label_set 

net_integrate.forward()

print 'Cost time : ' + str(int(round(time.time() * 1000)) - TimeStart) + 'ms.'

prediction = net_integrate.blobs['output'].data
prediction = np.array(prediction)
prediction = prediction[0].argmax(axis=0)
del net_integrate

scoreFilename = '/score%04d.npy' % target
np.save(output_path + scoreFilename, prediction)
color_image = color_panel[prediction.ravel()].reshape((425,560,3))

resultFilename = '/result%04d.png' % target
cv2.imwrite(output_path + resultFilename, color_image)

scoreMatFilename = '/score%04d.mat' % target
sio.savemat(output_path + scoreMatFilename, { 'pixelClasses' : prediction });

