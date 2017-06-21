import matplotlib
matplotlib.use('Agg')

import os
import os.path
import gc
import sys
import time
import random
import scipy.misc
import numpy as np
import subprocess
import datetime
import math
import colorsys
import tensorflow as tf
import pylab as Plot

from PIL import Image, ImageOps
from os import listdir
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_similarity_score
from skimage import img_as_float


from matplotlib.mlab import PCA as mlabPCA
from matplotlib import pyplot as plt
NUM_CLASSES = 2
TEST_STR = ""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

################################################################################################################### TSNE PLOT https://lvdmaaten.github.io/tsne/

def printParams(listParams):
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	for i in xrange(1,len(sys.argv)):
		print listParams[i-1] + '= ' + sys.argv[i]
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	
def softmax(array):
	expa = np.exp(array)
	sumexp = np.sum(expa, axis=1)
	sumexp_repeat = np.repeat(sumexp, 2, axis=0).reshape((len(expa),2))
	soft_calc = np.divide(expa, sumexp_repeat)
	#print array[0], expa[0], sumexp_repeat[0], soft_calc[0]
	#print array[1], expa[1], sumexp_repeat[1], soft_calc[1]
	return soft_calc

def jaccardIndex(set1, set2):
	unionSet = set1.union(set2)
	intSet = set1.intersection(set2)

	jaccard = len(intSet)/float(len(unionSet))
	
	return jaccard

def WriteImageP2(array,output):
	print("Writing .PGM")
	print(output)
        fp = open(output,"w+")

        #New Header
        magic_number = "P2"
        rows    = int(array.shape[0])
        columns = int(array.shape[1])
        max_gray_level = int(np.amax(array))

        print magic_number
        print columns,rows
        print max_gray_level

        #Writing Header File    
        fp.write(str(magic_number+"\n"))
        fp.write(str(columns)+" ")
        fp.write(str(rows)+"\n")
        fp.write(str(max_gray_level)+"\n")

        #Writing Gray Values Pixels
        array = array.astype("uint16")

        #Save Image in File - Traditional Way
        for row in xrange(rows):
                for column in xrange(columns):
                        if column == (columns-1) :
                                fp.write(str(array[row][column]))
                        else:
                                fp.write(str(array[row][column])+" ")

                fp.write("\n")

        fp.close()
	return()


def read_image_P2_int16(filename):
	fp = open(filename,"r")

	# Read the type of PGM
	magic_number = (fp.readline()).split()

	if magic_number[0] != "P2":
		print "This is not a P2 image"
		return 0
		
	# Search for comments
	info = (fp.readline()).split()
	if info[0].startswith('#'):
		#print info
		info = fp.readline().split()

	# Read Width and Height
	width  = int(info[0])
	height = int(info[1])
	#print width, height

	# Read the Max Grey Level
	max_gray_level = (fp.readline()).split()
	#print max_gray_level

	# END THE HEADER

	#Create New Array
	img = np.empty([height, width],dtype="uint8") #row,column 

	#Save Image in Numpy Array
	for row in xrange(height):
		for column in xrange(width):
			raw = fp.readline()
			img[row][column] = raw

	##print np.bincount(img.astype(int).flatten())
	return img

##################################################################################################################### Data Utils

def normalizeImages(data, mean_full, std_full):
	data[:,:,:,0] = np.subtract(data[:,:,:,0], mean_full[0])
	data[:,:,:,1] = np.subtract(data[:,:,:,1], mean_full[1])
	data[:,:,:,2] = np.subtract(data[:,:,:,2], mean_full[2])
	
	data[:,:,:,0] = np.divide(data[:,:,:,0], std_full[0])
	data[:,:,:,1] = np.divide(data[:,:,:,1], std_full[1])
	data[:,:,:,2] = np.divide(data[:,:,:,2], std_full[2])


def computeImageMean(data,cropSize):

	mean_full = 0
	std_full = 0
	samples = 0

	img = data
	lim = int(cropSize/2)

	mean = np.zeros((cropSize,cropSize,3))
	means = np.zeros(3)
    
        print("Compute mean and std...")     
	for linha in xrange(lim,img.shape[0]-lim):
		for coluna in xrange(lim,img.shape[1]-lim):
			sub = img[linha-lim:linha+lim+1,coluna-lim:coluna+lim+1,:]
                    
			samples = samples + 1
			for k in xrange(0,3):
				mean[:,:,k] = mean[:,:,k] + sub[:,:,k]
                
	for k in xrange(0,3):
		means[k] = sum(sum(mean[:,:,k]))/(samples*cropSize*cropSize)

	stds = np.zeros(3)
	cont = 0

	for linha in xrange(lim,img.shape[0]-lim):
		for coluna in xrange(lim,img.shape[1]-lim):
			sub = img[linha-lim:linha+lim+1,coluna-lim:coluna+lim+1,:]
			for ch in xrange(0,3):
				matSum = sum(sum((sub[:,:,ch])))/(cropSize*cropSize)
				pot = math.pow((matSum - means[ch]),2)
				stds[ch] = stds[ch] + pot
  
	for ch in xrange(0,3):
		stds[ch] = math.sqrt(stds[ch]/(samples-1))

	return means, stds
	
def manipulateBorderArray(data, cropSize):
	mask = int(cropSize/2)

	h,w = len(data), len(data[0])
	crop_left = data[0:h,0:cropSize]
	crop_right = data[0:h,w-cropSize:w,:]
	crop_top = data[0:cropSize,0:w,:]
	crop_bottom = data[h-cropSize:h,0:w,:]

	mirror_left = np.fliplr(crop_left)
	mirror_right = np.fliplr(crop_right)
	flipped_top = np.flipud(crop_top)
	flipped_bottom = np.flipud(crop_bottom)

	h_new,w_new = h+mask*2, w+mask*2
	data_border = np.zeros((h_new, w_new, len(data[0][0])))
	#print data_border.shape
	data_border[mask:h+mask,mask:w+mask,:] = data

	data_border[mask:h+mask, 0:mask, :] = mirror_left[:, mask+1:, :]
	data_border[mask:h+mask, w_new-mask:w_new ,:] = mirror_right[:,0:mask,:]
	data_border[0:mask, mask:w+mask, :] = flipped_top[mask+1:, : ,:]
	data_border[h+mask:h+mask+mask, mask:w+mask, :] = flipped_bottom[0:mask, : ,:]

	data_border[0:mask, 0:mask, :] = flipped_top[mask+1:, 0:mask ,:]
	data_border[0:mask, w+mask:w+mask+mask, :] = flipped_top[mask+1:, w-mask:w ,:]
	data_border[h+mask:h+mask+mask, 0:mask, :] = flipped_bottom[0:mask, 0:mask ,:]
	data_border[h+mask:h+mask+mask, w+mask:w+mask+mask, :] = flipped_bottom[0:mask, w-mask:w ,:]

	#scipy.misc.imsave('C:\\Users\\Keiller\\Desktop\\outfile.jpg', data_border)
	return data_border


def loadImages(dataPath, instances, cropSize,type):
        #type - 0 - load train image 1 - load validation image
	images = []
	masks = []
        means = []
        stds = []

	for i in instances:
                print("Loading image " + dataPath+i)
		try:
			img = Image.open(dataPath+i+"/image.ppm")
			mask = read_image_P2_int16(dataPath+i+"/mascara.pgm")
		except IOError:
			print "Could not open file from ", dataPath

		img.load()
		imgFloat = manipulateBorderArray(img_as_float(img), cropSize)
		maskBinary = np.floor(img_as_float(mask)+0.5)
		##print np.bincount(maskBinary.astype(int).flatten())
		
		images.append(imgFloat)
		masks.append(maskBinary)
              
                if(type == 0):
                        mean,std = computeImageMean(imgFloat,cropSize)
                        means.append(mean)
                        stds.append(std)
	if(type == 0 or type == 2):
                return np.asarray(images), np.asarray(masks),np.asarray(means),np.asarray(stds)
        else:
                return np.asarray(images), np.asarray(masks)

def createDistributionsOverClasses(maskData, cropSize, isPurityNeeded=False, limitProcess=False, isDebug=False):
	mask = int(cropSize/2)
	classes = [[[] for i in range(0)] for i in range(NUM_CLASSES)]
	purityIndexes = [[[] for i in range(0)] for i in range(NUM_CLASSES)]
	off,h,w = maskData.shape
	total_pixel_amount = cropSize*cropSize
	count = 2*[0]
	pure = 2*[0]

	for i in xrange(off):
		for j in xrange(h):
			for k in xrange(w):
				currentClass = retrieveClass(maskData[i][j][k])
			
				purity = np.bincount(maskData[i,max(0,j-mask):min(h,j+mask+1),max(0,k-mask):min(w,k+mask+1)].astype(int).flatten())[currentClass]
				classes[currentClass].append((purity/float(total_pixel_amount), (i,j,k)))
				count[currentClass] += 1
					
				if purity/float(total_pixel_amount) >= 0.99:
					pure[currentClass] += 1
					purityIndexes[currentClass].append(len(classes[currentClass])-1)
					###################################### ease process
					if limitProcess == True and len(purityIndexes[0]) >= 5000 and len(purityIndexes[1]) >= 5000:
						if isDebug == True:
							for i in xrange(len(classes)):
								print 'Class ', str(i), str(len(classes[i]))
								print 'Pure Class ', str(i), str(len(purityIndexes[i]))
						return classes, purityIndexes

	if isDebug == True:
		for i in xrange(len(classes)):
			print 'Class ' + str(i) + ':: ' + str(len(classes[i])) + " == " + str(count[i])
			print 'Pure Class ' + str(i) + ':: ' + str(len(purityIndexes[i]))  + " == " + str(pure[i])

	return classes, purityIndexes


##################################################################################################################### Patch Utils



def createPatchesFromClassDistributionWithMinibatch(data, mask_data, classDistribution, cropSize,shuffleMajority,shuffleMinority):

	mask = int(cropSize/2)
	patches = []
	classes = []
		

	for j in shuffleMajority:
		curClass = 0
		curPos = j
		curPurity = classDistribution[curClass][curPos][0]
		curMap = classDistribution[curClass][curPos][1][0]
		curX = classDistribution[curClass][curPos][1][1]+mask
		curY = classDistribution[curClass][curPos][1][2]+mask
		curPatch = data[curMap,curX-mask:curX+mask+1, curY-mask:curY+mask+1, :]				
		patches.append(curPatch)
		curClass = retrieveClass(mask_data[curMap,curX-mask,curY-mask])
		classes.append(curClass)


	for j in shuffleMinority:
		curClass = 1
		curPos = j
		curPurity = classDistribution[curClass][curPos][0]
		curMap = classDistribution[curClass][curPos][1][0]
		curX = classDistribution[curClass][curPos][1][1]+mask
		curY = classDistribution[curClass][curPos][1][2]+mask
		curPatch = data[curMap,curX-mask:curX+mask+1, curY-mask:curY+mask+1, :]				
		patches.append(curPatch)
		curClass = retrieveClass(mask_data[curMap,curX-mask,curY-mask])
		classes.append(curClass)
			

	return np.asarray(patches), np.asarray(classes)

def createPatchesFromClassDistribution(data, mask_data, classDistribution, cropSize,shuffle):

	mask = int(cropSize/2)
	patches = []
	classes = []
		

	for j in shuffle:
		curClass = (1 if j >= len(classDistribution[0]) else 0)
		curPos = (j-len(classDistribution[0]) if j >= len(classDistribution[0]) else j)
		curPurity = classDistribution[curClass][curPos][0]
		curMap = classDistribution[curClass][curPos][1][0]
		curX = classDistribution[curClass][curPos][1][1]+mask
		curY = classDistribution[curClass][curPos][1][2]+mask
		curPatch = data[curMap,curX-mask:curX+mask+1, curY-mask:curY+mask+1, :]				
		patches.append(curPatch)
		curClass = retrieveClass(mask_data[curMap,curX-mask,curY-mask])
		classes.append(curClass)
			

	return np.asarray(patches), np.asarray(classes)


def createPatchesFromClassDistributionTest(data, mask_data, classDistribution, cropSize,shuffle):

	mask = int(cropSize/2)
	patches = []
	classes = []
		

	for j in shuffle:
		curClass = (1 if j >= len(classDistribution[0]) else 0)
		curPos = (j-len(classDistribution[0]) if j >= len(classDistribution[0]) else j)
		curPurity = classDistribution[curClass][curPos][0]
		curMap = classDistribution[curClass][curPos][1][0]
		curX = classDistribution[curClass][curPos][1][1]+mask
		curY = classDistribution[curClass][curPos][1][2]+mask
		curPatch = data[curMap,curX-mask:curX+mask+1, curY-mask:curY+mask+1, :]				
		patches.append(curPatch)
		curClass = retrieveClass(mask_data[curMap,curX-mask,curY-mask])
		classes.append(curClass)
			

	return np.asarray(patches), np.asarray(classes)


def dynamicCreatePatchesFromMap(data, maskData, cropSize):
        mask = int(cropSize/2)
        patches = []
        classes = []
        
        for j in xrange(mask,len(data)-mask):
                for k in xrange(mask,len(data[j])-mask):
                        ##print j-mask,j,j+mask+1, k-mask,k,k+mask+1
                        patch = data[j-mask:j+mask+1,k-mask:k+mask+1,:]
                        if len(patch) != cropSize or len(patch[0]) != cropSize:
                                print "Error Patch size not equal Mask Size", len(patch), len(patch[0])

                        patches.append(patch)
                        current_class = retrieveClass(maskData[j-mask][k-mask])
                        classes.append(current_class)

        #print("Dynamic creation")
        #print mask, len(data)-mask
        #print len(patches), len(classes)
        return np.asarray(patches), np.asarray(classes)


	
def dynamicCreatePatches(data, instances, cropSize, maskData=None):
	mask = int(cropSize/2)
	patches = []
	classes = []
	
	for i in xrange(len(instances)):
		curPurity = instances[i][0]
		curMap = instances[i][1][0]
		curX = instances[i][1][1]+mask
		curY = instances[i][1][2]+mask

		##print curPurity, curMap, curX, curY
		curPatch = data[curMap, curX-mask:curX+mask+1, curY-mask:curY+mask+1, :]
		##print curPatch.shape
		if (len(curPatch) != cropSize or len(curPatch[0]) != cropSize):
			print "Error: Current patch size ", len(curPatch), len(curPatch[0])
			
		patches.append(curPatch)
		if maskData is not None:
			curClass = retrieveClass(maskData[curMap, curX-mask, curY-mask])
			classes.append(curClass)

	return np.asarray(patches), np.asarray(classes)

def dynamicPredictionMap(map,prob,classDistribution,shuffle,all_predcs,softs_val):

	cont = 0
	for j in shuffle:
		curClass = (1 if j >= len(classDistribution[0]) else 0)
		curPos = (j-len(classDistribution[0]) if j >= len(classDistribution[0]) else j)
		curX = classDistribution[curClass][curPos][1][1]
		curY = classDistribution[curClass][curPos][1][2]
		map[curX][curY] = all_predcs[cont]
		prob.append([curX,curY,softs_val[cont][0],softs_val[cont][1]])
		cont = cont + 1

	return map


	
def createPredictionMap(outputPath, array, map):
	#array = np.reshape(np.asarray(all_predcs), (1000,1000), order="C")

	#WriteImageP2(array,outputPath)

	##print "array", np.bincount(array.astype(int).flatten())
	##print "all_predcs", np.bincount(np.asarray(all_predcs).astype(int).flatten())
        #print("\n\nSave image " + outputPath + '/map_6x3_' + str(map) + '.jpeg\n\n')
	scipy.misc.imsave(outputPath, array)

	'''img = Image.new("1", (1000,1000), "black")

	count = 2*[0]
	for i in xrange(1000):
		for j in xrange(1000):
			#print int(i/1000), int(i%1000)
			img.putpixel((i, j), int(all_predcs[i*1000+j]))
			count[int(all_predcs[i])] += 1
	
	img_ar = np.array(img)
	print "img_ar", np.bincount(np.asarray(img_ar).astype(int).flatten())
	print "count", count
	img.save(outputPath + 'map_' + str(map) + '_predMap_' + str(step) + '.jpeg')'''

def retrieveClass(val):

	if val == 1.0:
		current_class = 1
	elif val == 0.0:
		current_class = 0
	else:
		print("ERROR: mask value not binary ", val)

	return current_class





###############################################################################################################################################################
'''
	TensorFlow
'''

##################################################### network definition
def leakyReLU(x, alpha=0.1):
	return tf.maximum(alpha*x,x)

def _variable_on_cpu(name, shape, ini):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=ini, dtype=tf.float32)
	return var

def _variable_with_weight_decay(name, shape, ini, wd):
	var = _variable_on_cpu(name, shape, ini)
	#tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
	#tf.contrib.layers.xavier_initializer(dtype=tf.float32))
	#tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss') # multiply || mul
		tf.add_to_collection('losses', weight_decay)
	return var

def _max_pool(input, kernel, strides, name, pad='SAME', debug=False):
	pool = tf.nn.max_pool(input, ksize=kernel, strides=strides, padding=pad, name=name)
	if debug:
		pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

	return pool
	
def _batch_norm(input, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,  
				lambda: tf.contrib.layers.batch_norm(input, is_training=True, center=False, updates_collections=None, scope=scope+'_bn'),  
                lambda: tf.contrib.layers.batch_norm(input, is_training=False, center=False, updates_collections=None, scope=scope+'_bn', reuse=True)
				)

def _conv_layer(input, kernelShape, name, weightDecay, is_training, pad='SAME', strides=[1,1,1,1], batchNorm=True):
	with tf.variable_scope(name) as scope:
		weights = _variable_with_weight_decay('weights', shape=kernelShape, ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', kernelShape[-1], tf.constant_initializer(0.1))

		conv_op = tf.nn.conv2d(input, weights, strides, padding=pad)
		conv_op_add_bias = tf.nn.bias_add(conv_op, biases)

		if batchNorm == True:
			conv_act = leakyReLU(_batch_norm(conv_op_add_bias, is_training, scope=scope.name))
		else:
			conv_act = leakyReLU(conv_op_add_bias)

		return conv_act

	
def convNet_ICPR_11(x, dropout, is_training, cropSize, weightDecay):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, cropSize, cropSize, 3]) ## default: 25x25
	#print x.get_shape()
	
	conv1 = _conv_layer(x, [3,3,3,64], 'ft_conv1', weightDecay, is_training, pad='VALID')
	print("Conv")
	print(conv1.get_shape())
	pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
	print("pool")
	print(pool1.get_shape())

	conv2 = _conv_layer(pool1, [3,3,64,128], 'ft_conv2', weightDecay, is_training, pad='VALID')
	print("Conv")
	print(conv2.get_shape())
	pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')
	print("pool")
	print(pool2.get_shape())



	with tf.variable_scope('ft_fc1') as scope:
		reshape = tf.reshape(pool2, [-1, 1*1*128])
		weights = _variable_with_weight_decay('weights', shape=[1*1*128, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
		drop_fc1 = tf.nn.dropout(reshape, dropout)
		fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))
	
	# Fully connected layer 2
	with tf.variable_scope('ft_fc2') as scope:
		weights = _variable_with_weight_decay('weights', shape=[1024, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

		# Apply Dropout
		drop_fc2 = tf.nn.dropout(fc1, dropout)
		fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

	# Output, class prediction
	with tf.variable_scope('ft_fc3_logits') as scope:
		weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return fc2, logits
	
def convNet_ICPR_17(x, dropout, is_training, cropSize, weightDecay):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, cropSize, cropSize, 3]) ## default: 25x25
	#print x.get_shape()
	
	conv1 = _conv_layer(x, [3,3,3,64], 'ft_conv1', weightDecay, is_training, pad='VALID')
	print("Conv")
	print(conv1.get_shape())
	pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
	print("pool")
	print(pool1.get_shape())

	conv2 = _conv_layer(pool1, [3,3,64,128], 'ft_conv2', weightDecay, is_training, pad='VALID')
	print("Conv")
	print(conv2.get_shape())
	pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='ft_pool2', pad='VALID')
	print("pool")
	print(pool2.get_shape())

	conv3 = _conv_layer(pool2, [3,3,128,256], 'ft_conv3', weightDecay, is_training, pad='VALID')
	print("Conv")
	print(conv3.get_shape())
	pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='ft_pool2', pad='VALID')
	print("pool")
	print(pool3.get_shape())


	

	with tf.variable_scope('ft_fc1') as scope:
		reshape = tf.reshape(pool3, [-1, 1*1*256])
		weights = _variable_with_weight_decay('weights', shape=[1*1*256, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
		drop_fc1 = tf.nn.dropout(reshape, dropout)
		fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))
	
	# Fully connected layer 2
	with tf.variable_scope('ft_fc2') as scope:
		weights = _variable_with_weight_decay('weights', shape=[1024, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

		# Apply Dropout
		drop_fc2 = tf.nn.dropout(fc1, dropout)
		fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

	# Output, class prediction
	with tf.variable_scope('ft_fc3_logits') as scope:
		weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return fc2, logits

def convNet_ICPR_33(x, dropout, is_training, cropSize, weightDecay):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, cropSize, cropSize, 3]) ## default: 25x25
	#print x.get_shape()
	
	conv1 = _conv_layer(x, [4,4,3,64], 'ft_conv1', weightDecay, is_training, pad='VALID')
	pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
	print("Conv")
	print(conv1.get_shape())
	print("Pool")
	print(pool1.get_shape())

	
	conv2 = _conv_layer(pool1, [4,4,64,128], 'ft_conv2', weightDecay, is_training, pad='VALID')
	pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')
	print("Conv")
	print(conv2.get_shape())
	print("Pool")
	print(pool2.get_shape())

	
	conv3 = _conv_layer(pool2, [4,4,128,256], 'ft_conv3', weightDecay, is_training, pad='VALID')
	pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool3', pad='VALID')
	print("Conv")
	print(conv3.get_shape())
	print("Pool")
	print(pool3.get_shape())


	with tf.variable_scope('ft_fc1') as scope:
		reshape = tf.reshape(pool3, [-1, 1*1*256])
		weights = _variable_with_weight_decay('weights', shape=[1*1*256, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
		drop_fc1 = tf.nn.dropout(reshape, dropout)
		fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))
	
	# Fully connected layer 2
	with tf.variable_scope('ft_fc2') as scope:
		weights = _variable_with_weight_decay('weights', shape=[1024, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

		# Apply Dropout
		drop_fc2 = tf.nn.dropout(fc1, dropout)
		fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

	# Output, class prediction
	with tf.variable_scope('ft_fc3_logits') as scope:
		weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return fc2, logits


def convNet_ICPR_25(x, dropout, is_training, cropSize, weightDecay):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, cropSize, cropSize, 3]) ## default: 25x25
	#print x.get_shape()
	
	conv1 = _conv_layer(x, [4,4,3,64], 'ft_conv1', weightDecay, is_training, pad='VALID')
	pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
	
	conv2 = _conv_layer(pool1, [4,4,64,128], 'ft_conv2', weightDecay, is_training, pad='VALID')
	pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')
	
	conv3 = _conv_layer(pool2, [3,3,128,256], 'ft_conv3', weightDecay, is_training, pad='VALID')
	pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='ft_pool3', pad='VALID')

	with tf.variable_scope('ft_fc1') as scope:
		reshape = tf.reshape(pool3, [-1, 1*1*256])
		weights = _variable_with_weight_decay('weights', shape=[1*1*256, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
		drop_fc1 = tf.nn.dropout(reshape, dropout)
		fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))
	
	# Fully connected layer 2
	with tf.variable_scope('ft_fc2') as scope:
		weights = _variable_with_weight_decay('weights', shape=[1024, 1024], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

		# Apply Dropout
		drop_fc2 = tf.nn.dropout(fc1, dropout)
		fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

	# Output, class prediction
	with tf.variable_scope('ft_fc3_logits') as scope:
		weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return fc2, logits


	print(conv2.get_shape())
def convNet_ICPR_45(x, dropout, is_training, cropSize, weightDecay):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, cropSize, cropSize, 3]) ## default: 25x25
	#print x.get_shape()
	
	conv1 = _conv_layer(x, [4,4,3,128], 'ft_conv1', weightDecay, is_training, pad='VALID')
	#print("conv shape")
	#print(conv1.get_shape())

	pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
	#print("pool shape")	
	#print(pool1.get_shape())

	conv2 = _conv_layer(pool1, [4,4,128,192], 'ft_conv2', weightDecay, is_training, pad='VALID')
	#print("conv shape")
	#print(conv2.get_shape())

	pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')
	#print("pool shape")
	#print(pool2.get_shape())
	
	
	conv3 = _conv_layer(pool2, [4,4,192,256], 'ft_conv3', weightDecay, is_training, pad='VALID')
	#print("conv shape")
	#print(conv3.get_shape())


	pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='ft_pool3', pad='VALID')
	#print("pool shape")	
	#print(pool3.get_shape())


	conv4 = _conv_layer(pool3, [4,4,256,312], 'ft_conv4', weightDecay, is_training, pad='VALID')
	#print("conv shape")
	#print(conv4.get_shape())


	pool4 = _max_pool(conv4, kernel=[1, 2, 2, 1], stride
	print(conv2.get_shape())s=[1, 1, 1, 1], name='ft_pool4', pad='VALID')
	#print("pool shape")	
	#print(pool4.get_shape())

	with tf.variable_scope('ft_fc1') as scope:
		reshape = tf.reshape(pool4, [-1, 1*1*312])
		weights = _variable_with_weight_decay('weights', shape=[1*1*312, 96], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
		drop_fc1 = tf.nn.dropout(reshape, dropout)
		fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))
	
	# Fully connected layer 2
	with tf.variable_scope('ft_fc2') as scope:
		weights = _variable_with_weight_decay('weights', shape=[96, 96], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))

		# Apply Dropout
		drop_fc2 = tf.nn.dropout(fc1, dropout)
		fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

	# Output, class prediction
	with tf.variable_scope('ft_fc3_logits') as scope:
		weights = _variable_with_weight_decay('weights', [96, NUM_CLASSES], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return fc2, logits

def convNet_ICPR_57(x, dropout, is_training, cropSize, weightDecay):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, cropSize, cropSize, 3]) ## default: 25x25
	#print x.get_shape()
	
	conv1 = _conv_layer(x, [4,4,3,128], 'ft_conv1', weightDecay, is_training, pad='VALID')
	print("conv shape")
	print(conv1.get_shape())

	pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
	print("pool shape")	
	print(pool1.get_shape())

	conv2 = _conv_layer(pool1, [4,4,128,192], 'ft_conv2', weightDecay, is_training, pad='VALID')
	print("conv shape")
	print(conv2.get_shape())

	pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')
	print("pool shape")
	print(pool2.get_shape())
	
	
	conv3 = _conv_layer(pool2, [4,4,192,256], 'ft_conv3', weightDecay, is_training, pad='VALID')
	print("conv shape")
	print(conv3.get_shape())


	pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool3', pad='VALID')
	print("pool shape")	
	print(pool3.get_shape())


	conv4 = _conv_layer(pool3, [3,3,256,312], 'ft_conv4', weightDecay, is_training, pad='VALID')
	print("conv shape")
	print(conv4.get_shape())


	pool4 = _max_pool(conv4, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='ft_pool4', pad='VALID')
	print("pool shape")
	print(pool4.get_shape())



	with tf.variable_scope('ft_fc1') as scope:
		reshape = tf.reshape(pool4, [-1, 1*1*312])
		weights = _variable_with_weight_decay('weights', shape=[1*1*312, 96], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
		drop_fc1 = tf.nn.dropout(reshape, dropout)
		fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))
	
	# Fully connected layer 2
	with tf.variable_scope('ft_fc2') as scope:
		weights = _variable_with_weight_decay('weights', shape=[96, 96], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))

		# Apply Dropout
		drop_fc2 = tf.nn.dropout(fc1, dropout)
		fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

	# Output, class prediction
	with tf.variable_scope('ft_fc3_logits') as scope:
		weights = _variable_with_weight_decay('weights', [96, NUM_CLASSES], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

	return fc2, logits


def loss_def(logits, labels):
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def convNet_ICPR(x, keep_prob, is_training, cropSize, weightDecay):

	print("Cropsize " + str(cropSize))
	if(cropSize == 11):
		 _,logits = convNet_ICPR_11(x, keep_prob, is_training, cropSize, weightDecay)
	elif(cropSize == 17):
		 _,logits = convNet_ICPR_17(x, keep_prob, is_training, cropSize, weightDecay)
	elif(cropSize == 33):
		 _,logits = convNet_ICPR_33(x, keep_prob, is_training, cropSize, weightDecay)
	elif(cropSize == 25):
		 _,logits = convNet_ICPR_25(x, keep_prob, is_training, cropSize, weightDecay)
	elif(cropSize == 45):
		 _,logits = convNet_ICPR_45(x, keep_prob, is_training, cropSize, weightDecay)
	elif(cropSize == 57):
		 _,logits = convNet_ICPR_57(x, keep_prob, is_training, cropSize, weightDecay)
	elif(cropSize == 65):
		 _,logits = convNet_ICPR_65(x, keep_prob, is_training, cropSize, weightDecay)

	
	return _,logits

def drawGraphic(valuesFile,graphicFile):

	result = open(valuesFile,'r')

	training = []
	validation = []
	epoch = 0
	for line in result:
		l = line.rstrip()
		losses = l.split(';')
		training.append(float(losses[0]))
		validation.append(float(losses[1]))
		epoch+=1

	result.close()

	plt.clf()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot(training,'r--', label="Training")
	plt.plot(validation,'b--',label="Validation")

	training = np.asarray(training)
	validation = np.asarray(validation)

	#maxValue = np.amax(training) if np.amax(training) > np.amax(validation) else np.amax(validation)
	#minValue = np.amin(training) if np.amin(training) < np.amin(validation) else np.amin(validation)

	plt.yticks(np.arange(0,10,0.5))
	plt.xticks(np.arange(0,epoch,1))
	plt.legend(loc='best')

	plt.savefig(graphicFile)


def train(instance,dataPath,trainInstances,trainData,trainMask,validationData,validationMask,classes,purity,classesValidation,mean_full,std_full,lr_initial,batchSize,weightDecay,decayParam,cropSize,outputPath,display_step,val_inteval,epochs,countIter,useMinibatch,useValidation,keepTraining=False,isFullTraining=False):                 


	
        path = "models/" + str(instance) + "_model_6x3_" + str(cropSize)

        for name in trainInstances:
                path = path + "_" + name

	if useMinibatch == 1:
		path = path + "_minibatch"
        
	print("Model path: " + path)
	print("Number of images used to train: " + str(len(trainInstances)))
	print("Iteration: " + str(countIter))
	print("Epochs: " + str(epochs))
	

	resultPath = "results/maps/" + str(instance) + "_maps_6x3_" + str(cropSize) + "/train/"
	resultPath = outputPath + resultPath
	if os.path.exists(resultPath) != True:
		print("Creating folder: " + resultPath)
                os.makedirs(resultPath)

	trainings = ""	
	trainings = trainInstances[0]
	for i in xrange(1,len(trainInstances)):
		trainings = trainings + "-" + trainInstances[i]
	
	resultPath = resultPath + trainings
	
	if os.path.exists(resultPath) != True:
		print("Creating folder: " + resultPath)
                os.makedirs(resultPath)


	resultFile = resultPath + "/training-result.txt"
	graphicFile = resultPath + "/train-validation.png"       

	if os.path.isfile(resultFile) != True or countIter == 0:
		print("Creating file: " + resultFile)
       		result = open(resultFile,"w") 
		result.close()

	window = 1
	######################################## Network Parameters
	n_input = cropSize*cropSize*3 # RGB
	dropout = 0.5 # Dropout, probability to keep units

	######################################## tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	is_training = tf.placeholder(tf.bool, [], name='is_training')

	######## CONVNET
	#### ICPR
	_,logits = convNet_ICPR(x, keep_prob, is_training, cropSize, weightDecay)
	
	####################################### Define loss and optimizer
	loss = loss_def(logits, y)
          
	if countIter == 0:
		global_step = tf.Variable(0, name='global_step', trainable=False)
		lr = tf.train.exponential_decay(lr_initial, global_step, decayParam, 0.1, staircase=True) ##### original
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)
	else:
		global_step = tf.Variable(0, name='init_global_step', trainable=False)
		lr = tf.train.exponential_decay(lr_initial, global_step, decayParam, 0.1, staircase=True) ##### original
		optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

	###################################### Evaluate model
	correct = tf.nn.in_top_k(logits, y, 1)
	# Return the number of true entries
	acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
	pred = tf.argmax(logits, 1)
	soft = tf.nn.softmax(logits)

	# Initializing the variables || Add ops to save and restore all the variables.
	if countIter == 0:
		init = tf.initialize_all_variables()
		saver = tf.train.Saver()
	else:
		init = tf.initialize_variables([k for k in tf.all_variables() if k.name.startswith('init_')])
		saver = tf.train.Saver()
		saver_restore = tf.train.Saver([k for k in tf.all_variables() if k.name.startswith('ft_')])




	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		if countIter == 0 or keepTraining==False:
			sess.run(init)
		elif keepTraining==True: 
			if isFullTraining == False:
				if (countIter-1) > 1:
					sess.run(init)
					print('Restoring training Model from ' + outputPath+path + '_iteration_' + str(countIter-1) + ' ...')
					saver_restore.restore(sess, outputPath+ path + '_iteration_'+ str(countIter-1))
					print('...Done!')
				else:
					sess.run(init)
					print('Restoring training Model from ' + outputPath+path + ' ...')
					saver_restore.restore(sess, outputPath+path)
					print('...Done!')
			else:
				sess.run(init)
				print('Restoring training Model from ' + outputPath+path + '_final ...')
				saver_restore.restore(sess, outputPath+ path + '_final')
				print('...Done!')

                it = 0
	        holdon = 0
	        trackLoss = np.zeros([len(trainInstances)*10000*window*batchSize], dtype=np.float32)
         	epoch_mean = 0.0
	        epoch_cm_train = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
		batch_cm_train = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)

		#Tem de dividir por 2 porque tiraremos metade do batchsize de uma classe e metade da outra totalizando o batchsize que queremos
		if useMinibatch == 1:
			batchSize = int(batchSize/2)

 		epochs =  countIter+epochs+1
                for e in range(countIter,epochs):
                        
                        print("Epoch: " + str(e) + " / " + str(epochs-1))
                 
			if useMinibatch == 1:
				
				shuffleClass0 = np.asarray(random.sample(xrange(len(classes[0])), (len(classes[0]))))
				shuffleClass1 = np.asarray(random.sample(xrange(len(classes[1])), (len(classes[1]))))

				majorityShuffle = shuffleClass0
				minorityShuffle = shuffleClass1

				if len(shuffleClass0) < len(shuffleClass1):
					majorityShuffle = shuffleClass1
					minorityShuffle = shuffleClass0

				step1 = 0
				shuffleSize = len(majorityShuffle)
				epoch = int(shuffleSize/batchSize*2)

			else:
				shuffle = np.asarray(random.sample(xrange(len(classes[0]) + len(classes[1])), (len(classes[0]) + len(classes[1]))))
				shuffleSize = len(shuffle)
				epoch = int(shuffleSize/batchSize)
			
			for step in xrange(0,((shuffleSize/batchSize)+1 if shuffleSize%batchSize != 0 else (shuffleSize/batchSize))):

				if useMinibatch == 1:
					if len(shuffleClass0) < len(shuffleClass1):
						majorityShuffle = shuffleClass1
						minorityShuffle = shuffleClass0

					#Seleciona os indices da classe 0
					indsMajorityClass = majorityShuffle[step*batchSize:min(step*batchSize+batchSize, len(majorityShuffle))]
					sizeClasse1 = len(indsMajorityClass)
	
					indsMinorityClass = minorityShuffle[step1*sizeClasse1:min(step1*sizeClasse1+sizeClasse1, len(minorityShuffle))]
					
					if(len(indsMinorityClass) == 0):
						step1 = 0
						indsMinorityClass = minorityShuffle[step1*sizeClasse1:min(step1*sizeClasse1+sizeClasse1, len(minorityShuffle))]

					step1 += 1
					batch_x,batch_y = createPatchesFromClassDistributionWithMinibatch(trainData, trainMask, classes, cropSize,indsMajorityClass,indsMinorityClass)

				else:
                                	inds = shuffle[step*batchSize:min(step*batchSize+batchSize,shuffleSize)]
					batch_x,batch_y = createPatchesFromClassDistribution(trainData, trainMask, classes, cropSize,inds)
                                
				normalizeImages(batch_x, mean_full, std_full)
                                batch_x = np.reshape(batch_x,(-1,n_input))
                                _, batch_loss, batch_logits, batch_correct, batch_predcs = sess.run([optimizer, loss,logits,acc_mean,pred], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, is_training: True})

	        	        if math.isnan(batch_loss):
		                       print batch_x.shape
			               print batch_y.shape
				       print batch_y
				       print batch_loss
				       print batch_correct
				       print batch_predcs
				       print batch_predcs.shape
				       print batch_logits
				       print batch_logits.shape
				       return

		                epoch_mean += batch_correct
			        for j in range(len(batch_predcs)):
			               epoch_cm_train[batch_y[j]][batch_predcs[j]] += 1		
			
                                trackLoss[step%(epoch*window*batchSize)] = batch_loss
                                
         		        for j in range(len(batch_predcs)):
         				batch_cm_train[batch_y[j]][batch_predcs[j]] += 1

	        		_sum = 0.0
               			for i in xrange(len(batch_cm_train)):
					_sum += (batch_cm_train[i][i]/float(np.sum(batch_cm_train[i])) if np.sum(batch_cm_train[i]) != 0 else 0)

                                if(step%1000 == 0 and step > 0):


            			       print("Iter " + str(step) + " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
	        		        	" Absolut Right Pred= " + str(int(batch_correct)) +
		         		        " Overall Accuracy= " + "{:.4f}".format(batch_correct/float(len(batch_y))) +
			        	        " Normalized Accuracy= " + "{:.4f}".format(_sum/float(NUM_CLASSES)) +
				                " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
			               )

			if useValidation == 1:
				print("Validation...")
				if useMinibatch == 1:
					batch = batchSize*2
				else:
					batch = batchSize

				validation(sess,countIter,resultFile,validationData,validationMask,mean_full,std_full,classesValidation,batch,cropSize,pred,acc_mean,soft,x,y,keep_prob,is_training,batch_loss,loss)
				drawGraphic(resultFile,graphicFile)

			if e > 0 and e < epochs-1:
				print("Saving model: " + outputPath+path+'_iteration_'+str(e))
				saver.save(sess, outputPath+path+'_iteration_'+str(e))
			elif e == 0: 
				print("Saving model: " + outputPath+path)
				saver.save(sess, outputPath+path)
			
				


	print("Optimization Finished!")

	print("Saving model: " + outputPath+path+'_final')
	saver.save(sess, outputPath+path + '_final')		
                
	tf.reset_default_graph()        
        
	return np.mean(trackLoss)

                
        
def extractFeatures(patches, batchSize, weightDecay, cropSize, outputPath, countIter):
	######################################## Network Parameters
	n_input = cropSize*cropSize*3 # RGB
	dropout = 0.5 # Dropout, probability to keep units

	######################################## tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	is_training = tf.placeholder(tf.bool, [], name='is_training')

	######## CONVNET
	#### ICPR
	fc2, logits = convNet_ICPR_25(x, keep_prob, is_training, cropSize, weightDecay)

	# Initializing the variables || Add ops to save and restore all the variables.
	saver_restore = tf.train.Saver([k for k in tf.all_variables() if k.name.startswith('ft')])

	features = np.empty([len(patches),1024], dtype=np.float64)
	all_logits = np.empty([len(patches),NUM_CLASSES], dtype=np.float64)
	
	# Launch the graph
	with tf.Session() as sess:
		if countIter >= 1:
			print('Restoring Model from ' + outputPath+'model_final_'+str(countIter) + '...')
			saver_restore.restore(sess, outputPath+'model_final_'+str(countIter))
			print('... Done!')
		else:
			print('Restoring Model from ' + outputPath+'fullTraining_model_final')
			#saver_restore.restore(sess, outputPath+'fullTraining_modelDA_final')
			saver_restore.restore(sess, outputPath+'fullTraining_model_final')
			print('... Done!')

		# Keep training until reach max iterations
		for i in xrange(0,(len(patches)/batchSize)):
			bx = np.reshape(patches[i*batchSize:min(i*batchSize+batchSize, len(patches))], (-1, n_input))
			### fake classes
			by = np.zeros(len(bx), dtype=np.uint8)

			# extract features
			_fc2, _logits = sess.run([fc2, logits], feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})

			features[i*batchSize:min(i*batchSize+batchSize, len(patches)),:] = _fc2
			all_logits[i*batchSize:min(i*batchSize+batchSize, len(patches)),:] = _logits

	tf.reset_default_graph()
	### validation for NANs
	if np.any(np.isnan(features)):
		print bcolors.FAIL + 'Features NAN == ' + str(np.any(np.isnan(features))) + bcolors.ENDC
	

	return features, all_logits


def extractFeaturesFromClassDistribution(data, classDistribution, batchSize, weightDecay, cropSize, outputPath, countIter, mean_full, std_full, maskData=None):
	######################################## Network Parameters
	n_input = cropSize*cropSize*3 # RGB
	dropout = 0.5 # Dropout, probability to keep units

	######################################## tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	is_training = tf.placeholder(tf.bool, [], name='is_training')

	######## CONVNET
	#### ICPR
	fc2, logits = convNet_ICPR_25(x, keep_prob, is_training, cropSize, weightDecay)

	# Initializing the variables || Add ops to save and restore all the variables.
	saver_restore = tf.train.Saver([k for k in tf.all_variables() if k.name.startswith('ft')])

	if len(classDistribution) == 2:	
		totalLength = int(len(classDistribution[0]) + len(classDistribution[1]))
		total = classDistribution[0] + classDistribution[1]
	else:
		totalLength = int(len(classDistribution))
		total = classDistribution

	print 'totalLength', totalLength

	features = np.empty([totalLength, 1024], dtype=np.float16)
	all_logits = np.empty([totalLength, NUM_CLASSES], dtype=np.float16)
	all_classes = np.empty([totalLength], dtype=np.uint8)
	
	# Launch the graph
	count = 0
	with tf.Session() as sess:
		if countIter >= 1:
			print('Restoring Model from ' + outputPath+'model_final_'+str(countIter) + '...')
			saver_restore.restore(sess, outputPath+'model_final_'+str(countIter))
			print('... Done!')
		else:
			print('Restoring Model from ' + outputPath+'fullTraining_model_final')
			#saver_restore.restore(sess, outputPath+'fullTraining_modelDA_final')
			saver_restore.restore(sess, outputPath+'fullTraining_model_final')
			print('... Done!')

		# Keep training until reach max iterations
		for i in xrange(0,(totalLength/batchSize)+1):
			bxx, by = dynamicCreatePatches(data, total[i*batchSize:min(i*batchSize+batchSize, totalLength)], cropSize, (None if maskData == None else maskData[i*batchSize:min(i*batchSize+batchSize, totalLength)]))
			normalizeImages(bxx, mean_full, std_full)
			bx = np.reshape(bxx, (-1, n_input))
			if maskData != None:
				all_classes[i*batchSize:min(i*batchSize+batchSize, totalLength),:] = by
			### fake classes
			#by = np.zeros(len(bx), dtype=np.uint8)

			count += len(bx)
			# extract features
			_fc2, _logits = sess.run([fc2, logits], feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})

			features[i*batchSize:min(i*batchSize+batchSize, totalLength),:] = _fc2
			all_logits[i*batchSize:min(i*batchSize+batchSize, totalLength),:] = _logits

	tf.reset_default_graph()
	#print 'total features extracted', count
	### validation for NANs
	if np.any(np.isnan(features)):
		print bcolors.FAIL + 'Features NAN == ' + str(np.any(np.isnan(features))) + bcolors.ENDC

	return features, all_logits, all_classes

	
def dynamicExtractFeaturesFromClassDistribution(data, mask, classDistribution, training_classes, training_patches_feats, batchSize, weightDecay, cropSize, outputPath, countIter, mean_full, std_full):
	shuffle = np.asarray(random.sample(xrange(len(classDistribution[0])), len(classDistribution[1])))
	total = [classDistribution[0][i] for i in shuffle] + classDistribution[1]
	totalLength = int(len(total))
	bins = 5
	
	print totalLength
	
	selected_feats = []
	selected_logits = []
	selected_pixels = []
	
	for i in xrange(bins):
		all_feats, all_logits = extractFeaturesFromClassDistribution(data, total[i*(int(totalLength/bins)):i*(int(totalLength/bins))+(int(totalLength/bins))+(0 if i != bins-1 else (int(totalLength%bins)))], batchSize, weightDecay, cropSize, outputPath, countIter, mean_full, std_full)
		print all_feats.shape, all_logits.shape
		nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto', n_jobs=12).fit(all_feats)
		distances, indices = nbrs.kneighbors(training_patches_feats)
		print distances.shape, indices.shape
		
		for index in xrange(len(indices)):
			for node in xrange(len(indices[index])):
				selected_feats.append(all_feats[indices[index][node]])
				selected_logits.append(all_logits[indices[index][node]])
				selected_pixels.append(total[indices[index][node]])
		gc.collect()

	'''selected_feats = (np.concatenate((all_distances,distances), axis=1) if i > 0 else distances)
	selected_logits = (np.concatenate((all_indices,distances), axis=1) if i > 0 else indices)
	print len(selected_feats)
	print len(selected_logits)
	print len(selected_pixels)
	
	print selected_feats[0]
	print selected_logits[0]
	print selected_pixels[0]'''
	
	return dynamicSelectNewTrainingSet(data, mask, selected_pixels, selected_feats, selected_logits, training_classes, training_patches_feats, cropSize)


def validation(sess,countIter,resultFile,validationData,validationMask,mean_full,std_full,classeDistribution,batchSize,cropSize,pred,acc_mean,soft,x,y,keep_prob,is_training,training_loss,loss,isValidation=True):


	# Launch the graph

	n_input = cropSize*cropSize*3
	all_true_count = np.zeros((len(validationData)), dtype=np.uint32)
	all_cm_test = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
	all_kappa = np.zeros((len(validationData)), dtype=np.float32)
	all_size = 0.0

        epoch_mean = 0.0
	epoch_cm_train = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
	batch_cm_train = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
	                       
	shuffle = np.arange(len(classeDistribution[0]) + len(classeDistribution[1]))

        epoch = int(len(shuffle)/batchSize)
	classes = []
	all_predcs = []
	probs = []
	cm_test = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
	true_count = 0.0
	map = np.zeros((1000,1000))

        for step in xrange(0,((len(shuffle)/batchSize)+1 if len(shuffle)%batchSize != 0 else (len(shuffle)/batchSize))):

        	inds = shuffle[step*batchSize:min(step*batchSize+batchSize, len(shuffle))]
                batch_x,batch_y = createPatchesFromClassDistributionTest(validationData, validationMask, classeDistribution, cropSize,inds)
                normalizeImages(batch_x, mean_full, std_full)
                batch_x = np.reshape(batch_x,(-1,n_input))
		preds_val, acc_mean_val, softs_val, loss_val = sess.run([pred, acc_mean,soft,loss], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})


		true_count += acc_mean_val
		all_predcs = np.concatenate((all_predcs,preds_val))
		classes = np.concatenate((classes,batch_y))

		for j in range(len(preds_val)):
			cm_test[batch_y[j]][preds_val[j]] += 1
			all_cm_test[batch_y[j]][preds_val[j]] += 1


		if isValidation == False:
			dynamicPredictionMap(map,probs,classeDistribution,inds,preds_val,softs_val)

	if isValidation == False:

		print("Creating prediction map for cropsize " + str(cropSize) + " and instance " + str(instance))
		createPredictionMap(outputPath + path, map, testInstance)
			
		print("Saving probability for cropsize " + str(cropSize) + " and instance " + str(instance))
		np.save("output/probability/" + str(instance) + "_probs_" + str(cropSize) + "_" + testInstance + ".npy",probs)
	
	
		_sum = 0.0
		for z in xrange(len(cm_test)):
			_sum += (cm_test[z][z]/float(np.sum(cm_test[z])) if np.sum(cm_test[z]) != 0 else 0)

		cur_kappa = cohen_kappa_score(classes, np.asarray(all_predcs))
		print(bcolors.OKGREEN + "---- Iter " + str(iteration) +
			" -- Time " + str(datetime.datetime.now().time()) + " -- Execution Time " + str(time.time() - start_time) +
			" -- Test: Overall Accuracy= " + "{:.6f}".format(true_count/float(len(classes))) +
			" Normalized Accuracy= " + "{:.6f}".format(_sum/float(NUM_CLASSES)) +
			" Kappa= " + "{:.4f}".format(cur_kappa) +
			" Confusion Matrix= " + np.array_str(cm_test).replace("\n", "") +
			bcolors.ENDC)
	else:

	
		print("Loading file: " + resultFile)
        	result = open(resultFile,"a") 


		print("Training loss: " + str("{:.6f}".format(training_loss)) + " Validation loss " + str("{:.6f}".format(loss_val)))
		result.write(str("{:.6f}".format(training_loss)) + ";" + str("{:.6f}".format(loss_val)) + "\n")
		result.close()

	'''
	result.write("---- Iter " + str(iteration) +
		" -- Time " + str(datetime.datetime.now().time()) + " -- Execution Time " + str(time.time() - start_time) +
		" -- Test: Overall Accuracy= " + "{:.6f}".format(true_count/float(len(classes))) +
		" Normalized Accuracy= " + "{:.6f}".format(_sum/float(NUM_CLASSES)) +
		" Kappa= " + "{:.4f}".format(cur_kappa) +
		" Confusion Matrix= " + np.array_str(cm_test).replace("\n", "") + "\n")
	'''
		

def test(instance,dataPath,trainInstances,testInstances,countIter, cropSize, batchSize, weightDecay, outputPath, iteration, mean_full, std_full,useMinibatch, isFullTraining=False, isDataAugmentation=False):


	trainings = ""	

        model_path = "models/" + str(instance) + "_model_6x3_" + str(cropSize) 

        for name in trainInstances:
                model_path = model_path + "_" + name

	if useMinibatch == 1:
		model_path = model_path + "_minibatch"

	print("Model path: " + model_path)
	print("Number of images used to test: " + str(len(testInstances)))
	print("Iteration: " + str(countIter))

	

		

	resultPath = "results/maps/" + str(instance) + "_maps_6x3_" + str(cropSize) + "/test/"
	resultPath = outputPath + resultPath
	if os.path.exists(resultPath) != True:
		print("Creating folder: " + resultPath)
                os.makedirs(resultPath)

	tests = ""	
	tests = testInstances[0]
	for i in xrange(1,len(testInstances)):
		tests = tests + "-" + testInstances[i]
	
	resultPath = resultPath + tests
	


	if os.path.exists(resultPath) != True:
		print("Creating folder: " + resultPath)
                os.makedirs(resultPath)



	

	start_time = time.time()
	######################################## Network Parameters
	n_input = cropSize*cropSize*3 # RGB
	dropout = 0.5 # Dropout, probability to keep units

	######################################## tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	is_training = tf.placeholder(tf.bool, [], name='is_training')

	######## CONVNET
	#### ICPR
	_,logits = convNet_ICPR(x, keep_prob, is_training, cropSize, weightDecay)
	
	####################################### Define loss and optimizer
	loss = loss_def(logits, y)

	###################################### Evaluate model
	correct = tf.nn.in_top_k(logits, y, 1)
	# Return the number of true entries
	acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
	pred = tf.argmax(logits, 1)
	soft = tf.nn.softmax(logits)
	#correct_pred = tf.equal(pred, tf.argmax(y, 1))
	#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables || Add ops to save and restore all the variables.
	saver = tf.train.Saver([k for k in tf.all_variables() if k.name.startswith('ft')])
	
	resultFile = resultPath + "/test-result.txt"
	predictionPath = resultPath + "/prediction"
	probabilityPath = outputPath + "results/probability/"  + str(instance) + "_probs_" + str(cropSize)

	if isFullTraining == False:
		if countIter > 0:
			model_path = model_path +'_iteration_'+str(countIter)
			resultFile = resultPath + "/test-result"+'-iteration-'+str(countIter)+".txt"
			predictionPath = resultPath + "/prediction"+'-iteration-'+str(countIter)
			probabilityPath = outputPath + "results/probability/"  + str(instance) + "_probs_" + str(cropSize) + '_iteration_'+str(countIter)
	elif isFullTraining == True:
		model_path = model_path +'_final'
		resultFile = resultPath + "/test-result"+"-final.txt"
		predictionPath = resultPath + "/prediction-final"
		probabilityPath = outputPath + "results/probability/"  + str(instance) + "_probs_" + str(cropSize) + "_final"
	

	
       	

	model_path = outputPath + model_path
	

	print("Model location " + model_path)
	print("Result file location " + resultFile)
	print("Map location " + predictionPath)
	print("Probability location " + probabilityPath)

	
	result = open(resultFile,"w") 
	
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

         	saver.restore(sess, model_path)

		for testInstance in testInstances:

                	testData, testMask = loadImages(dataPath, [testInstance], cropSize,1)

			classDistributionFile = outputPath + "/classDistribution/classe_" + tests + ".npy"
			purityFile = outputPath + "/classDistribution/purity_" + tests + ".npy"
				
			if os.path.exists(classDistributionFile) != True or os.path.exists(purityFile) != True:
				classeDistribution,purity = createDistributionsOverClasses(testMask, cropSize, isPurityNeeded=False, limitProcess=False, isDebug=True)
				np.save(classDistributionFile,classeDistribution)
				np.save(purityFile,purity)

			else:
				print("Loading test class")
				classeDistribution = np.load(classDistributionFile)
	

	                # Launch the graph
	                all_true_count = np.zeros((len(testData)), dtype=np.uint32)
	                all_cm_test = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
	                all_kappa = np.zeros((len(testData)), dtype=np.float32)
	                all_size = 0.0

         		epoch_mean = 0.0
	        	epoch_cm_train = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
		        batch_cm_train = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
	                       
			#shuffle = np.asarray(random.sample(xrange(len(classeDistribution[0]) + len(classeDistribution[1])), (len(classeDistribution[0]) + len(classeDistribution[1]))))
			shuffle = np.arange(len(classeDistribution[0]) + len(classeDistribution[1]))

         	        epoch = int(len(shuffle)/batchSize)
			classes = []
			all_predcs = []
			probs = []
			cm_test = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.uint32)
			true_count = 0.0
			mapImage = np.zeros((1000,1000))

         	        for step in xrange(0,((len(shuffle)/batchSize)+1 if len(shuffle)%batchSize != 0 else (len(shuffle)/batchSize))):

                                inds = shuffle[step*batchSize:min(step*batchSize+batchSize, len(shuffle))]
                                batch_x,batch_y = createPatchesFromClassDistributionTest(testData, testMask, classeDistribution, cropSize,inds)
                                normalizeImages(batch_x, mean_full, std_full)
                                batch_x = np.reshape(batch_x,(-1,n_input))
				preds_val, acc_mean_val, softs_val = sess.run([pred, acc_mean,soft], feed_dict={x: batch_x, y: batch_y, keep_prob: 1., is_training: False})

				true_count += acc_mean_val
				all_predcs = np.concatenate((all_predcs,preds_val))
				classes = np.concatenate((classes,batch_y))
				dynamicPredictionMap(mapImage,probs,classeDistribution,inds,preds_val,softs_val)

				for j in range(len(preds_val)):
					cm_test[batch_y[j]][preds_val[j]] += 1
					all_cm_test[batch_y[j]][preds_val[j]] += 1

			print("Creating prediction map for cropsize " + str(cropSize) + " and instance " + str(instance))
			createPredictionMap(predictionPath + "_" + testInstance + ".png", mapImage, testInstance)
			
			print("Saving probability for cropsize " + str(cropSize) + " and instance " + str(instance))
			np.save(probabilityPath + "_" + testInstance + ".npy",probs)
		
			_sum = 0.0
			for z in xrange(len(cm_test)):
				_sum += (cm_test[z][z]/float(np.sum(cm_test[z])) if np.sum(cm_test[z]) != 0 else 0)

			cur_kappa = cohen_kappa_score(classes, np.asarray(all_predcs))
			print(bcolors.OKGREEN + "---- Iter " + str(iteration) +
				" -- Time " + str(datetime.datetime.now().time()) + " -- Execution Time " + str(time.time() - start_time) +
				" -- Test: Overall Accuracy= " + "{:.6f}".format(true_count/float(len(classes))) +
				" Normalized Accuracy= " + "{:.6f}".format(_sum/float(NUM_CLASSES)) +
				" Kappa= " + "{:.4f}".format(cur_kappa) +
				" Confusion Matrix= " + np.array_str(cm_test).replace("\n", "") +
				bcolors.ENDC)

			result.write("---- Iter " + str(iteration) +
				" -- Time " + str(datetime.datetime.now().time()) + " -- Execution Time " + str(time.time() - start_time) +
				" -- Test: Overall Accuracy= " + "{:.6f}".format(true_count/float(len(classes))) +
				" Normalized Accuracy= " + "{:.6f}".format(_sum/float(NUM_CLASSES)) +
				" Kappa= " + "{:.4f}".format(cur_kappa) +
				" Confusion Matrix= " + np.array_str(cm_test).replace("\n", "") + "\n")

				                			

	tf.reset_default_graph()
        result.close()
	

	
'''
        python cnn_knn_dynamic_jefersson.py /media/tensorflow/coffee/dataset/ /media/tensorflow/coffee/output/ 7_5,8_6,8_7 7_7,7_8,8_5 7_6,8_6,9_5 25 500 0.01 0.005 200 100

'''
def main():


	listParams = ['dataPath', 'outputPath(for model, images, etc)', 'trainInstances','validationInstances', 'testInstances', 'cropSize','learningRate', 'weightDecay', 'batchSize', 'nIter','epochs','trainModel','instance','useMinibatch','useValidation']




	if len(sys.argv) < len(listParams)+1:
		sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(listParams))
	printParams(listParams)

	#training images path
	index = 1
	dataPath = sys.argv[index]

	#output path
	index = index + 1
	outputPath = sys.argv[index]

	#training instances
	index = index + 1
	trainings = sys.argv[index]
	trainInstances = sys.argv[index].split(',')

	#validation instances
	index = index + 1
	validations = sys.argv[index]
	validationInstances = sys.argv[index].split(',')

	#test instances
	index = index + 1
	testInstances = sys.argv[index].split(',')
	
	#cropsize
	index = index + 1
	cropSize = int(sys.argv[index])
	        
	# NETWORK Parameters
	index = index + 1
	lr_initial = float(sys.argv[index])

	index = index + 1
	weightDecay = float(sys.argv[index])

	index = index + 1
	batchSize = int(sys.argv[index])

	index = index + 1
	countIter = int(sys.argv[index])

	index = index + 1
	epochs = int(sys.argv[index])

	index = index + 1
	train_model = int(sys.argv[index])

	index = index + 1
	instance = int(sys.argv[index])

	index = index + 1
	useMinibatch = int(sys.argv[index])
	
	index = index + 1
	useValidation = int(sys.argv[index])

	display_step = 50
	epoch_number = 5000
	val_inteval = 5000
	historyWindow = 10
		
	mean_file = outputPath + "/mean_std/" + str(instance) + "_mean_full.npy"
	std_file = outputPath + "/mean_std/" + str(instance) + "_std_full.npy"	

	if os.path.isfile(mean_file) != True or os.path.isfile(std_file) != True:
		trainData,trainMask,means,stds = loadImages(dataPath,trainInstances, cropSize,0)
		mean_full = np.asarray([0,0,0])
		std_full = np.asarray([0,0,0])
	
		for i in xrange(len(means)):
			mean_full = mean_full + means[i]
			std_full = std_full + stds[i]

		mean_full = mean_full/len(trainInstances)
		std_full = std_full/len(trainInstances)
		np.save(mean_file,mean_full)
		np.save(std_file,std_full)
	else:

		print("Loading mean and std")
		if train_model == 1:
	        	trainData,trainMask = loadImages(dataPath,trainInstances, cropSize,1)
		mean_full = np.load(mean_file)
		std_full = np.load(std_file)


	if(train_model == 1):

		if useMinibatch == 1:
			print("Using minibatch balance")
		else:
			print("Not using minibatch balance")

		if useValidation == 1:

			print("Using validation")
			print("Loading validation images")
	        	validationData,validationMask = loadImages(dataPath,validationInstances, cropSize,1)
			validationClassDistributionFile = outputPath + "/classDistribution/classe_" + validations + ".npy"
			validationPurityFile = outputPath + "/classDistribution/purity_" + validations + ".npy"

		        if os.path.exists(validationClassDistributionFile) != True:
		        	validationClasses,validationPurity = createDistributionsOverClasses(validationMask, cropSize, isPurityNeeded=False, limitProcess=False, isDebug=True)
				np.save(validationClassDistributionFile,validationClasses)
				np.save(validationPurityFile,validationPurity)

			else:
				print("Loading validation class distribution")
				classesValidation = np.load(validationClassDistributionFile)

		else:
			print("Not using validation")
			validationData = []
			validationMask = []
			classesValidation = []


			
		classDistributionFile = outputPath + "/classDistribution/classe_" + trainings + ".npy"
		purityFile = outputPath + "/classDistribution/purity_" + trainings + ".npy"
				
	        if os.path.exists(classDistributionFile) != True or os.path.exists(purityFile) != True:
	        	classes,purity = createDistributionsOverClasses(trainMask, cropSize, isPurityNeeded=False, limitProcess=False, isDebug=True)
			np.save(classDistributionFile,classes)
			np.save(purityFile,purity)

		else:
			print("Loading training class distribution and purity index")
			classes = np.load(classDistributionFile)
			purity = np.load(purityFile)


        	train(instance,dataPath,trainInstances,trainData,trainMask,validationData,validationMask,classes,purity,classesValidation,mean_full,std_full,lr_initial,batchSize,weightDecay,50000,cropSize,outputPath,display_step,val_inteval,epochs,countIter,useMinibatch,useValidation,keepTraining=True,isFullTraining=True)
	else:
		mean_full = np.load(mean_file)
		std_full = np.load(std_file)
		
	print("FULL TEST...")
	test(instance,dataPath,trainInstances, testInstances,countIter,cropSize, batchSize, weightDecay, outputPath, 'full', mean_full, std_full,useMinibatch, isFullTraining=True)
	print("...Done!")
	

if __name__ == "__main__":
    main()

