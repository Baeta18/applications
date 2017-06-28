import numpy as np 
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import input_data
from PIL import Image, ImageOps
import math
from skimage import img_as_float
import scipy.misc

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

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



def plotNNFilter(units,layer_number,label):
    #filters = units.shape[3]
    filters = 20
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    print("Total columns " + str(n_columns) + " rows " + str(n_rows))
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i],interpolation="nearest")

    filter_path = "/media/tensorflow/coffee/output/filters/weights_layer_" + str(layer_number) + "_label_" + str(label) + ".png"
    print("Saving image at: " + filter_path)
    plt.savefig(filter_path)

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

def retrieveClass(val):

	if val == 1.0:
		current_class = 1
	elif val == 0.0:
		current_class = 0
	else:
		print("ERROR: mask value not binary ", val)

	return current_class

tf.reset_default_graph()
outputPath = "/media/tensorflow/coffee/output/"
mean_file = outputPath + "/mean_std/5_mean_full.npy"
std_file = outputPath + "/mean_std/5_std_full.npy"	

trainData,trainMask = loadImages("/media/tensorflow/coffee/dataset/",["8_7"], 41,1)
print("Loading mean and std")
mean_full = np.load(mean_file)
std_full = np.load(std_file)

img = trainData[0]
mask = trainMask[0]

'''
print("Data shape")
print(trainData.shape)
print("Mask shape")
print(trainMask.shape)
'''



#patch = img[480:521,480:521,:]
#label = retrieveClass(mask[500][500])

patch = img[470:511,130:171,:]
label = 1
print("Label " + str(label))



img_path = "/media/tensorflow/coffee/output/filters/image_label_" + str(label) + ".png"
scipy.misc.imsave(img_path, patch)

batch_x = [patch]
batch_y = [label]

NUM_CLASSES = 2
n_input = 41*41*3 
weightDecay = 0.005
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
is_training = tf.placeholder(tf.bool, [], name='is_training')
dropout = 0.5 # Dropout, probability to keep units


x = tf.reshape(x, shape=[-1, 41, 41, 3]) ## default: 25x25
conv1 = _conv_layer(x, [5,5,3,128], 'ft_conv1', weightDecay, is_training, pad='SAME')
pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')
conv2 = _conv_layer(pool1, [4,4,128,192], 'ft_conv2', weightDecay, is_training, pad='SAME')
pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')
conv3 = _conv_layer(pool2, [3,3,192,256], 'ft_conv3', weightDecay, is_training, pad='SAME')
pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool3', pad='VALID')
conv4 = _conv_layer(pool3, [3,3,256,312], 'ft_conv4', weightDecay, is_training, pad='SAME')
pool4 = _max_pool(conv4, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool4', pad='VALID')

with tf.variable_scope('ft_fc1') as scope:
	reshape = tf.reshape(pool4, [-1, 2*2*312])
	weights = _variable_with_weight_decay('weights', shape=[2*2*312, 96], ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32), wd=weightDecay)
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


correct = tf.nn.in_top_k(logits, y, 1)
# Return the number of true entries
acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))


model_path = "/media/tensorflow/coffee/output/models/5_model_6x3_4_blocks_41_8_7_8_5_9_5_7_5_7_7_8_6_final"
saver = tf.train.Saver([k for k in tf.all_variables() if k.name.startswith('ft')])
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver.restore(sess, model_path)

testAccuracy = sess.run(acc_mean, feed_dict={x:batch_x,y:batch_y, keep_prob:1.,is_training: False})
print(testAccuracy)


print("test accuracy %g"%(testAccuracy))


units = sess.run(conv1,feed_dict={x:batch_x,keep_prob:1.0,is_training: False})
#units,layer,label
plotNNFilter(units,1,label)

units = sess.run(conv2,feed_dict={x:batch_x,keep_prob:1.0,is_training: False})
plotNNFilter(units,2,label)

units = sess.run(conv3,feed_dict={x:batch_x,keep_prob:1.0,is_training: False})
plotNNFilter(units,3,label)

units = sess.run(conv4,feed_dict={x:batch_x,keep_prob:1.0,is_training: False})
plotNNFilter(units,4,label)



'''
print("Load dataset")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

init = tf.initialize_all_variables()
sess.run(init)

x = tf.placeholder(tf.float32, [None, 784],name="x-in")
true_y = tf.placeholder(tf.float32, [None, 10],name="y-in")
keep_prob = tf.placeholder("float")

x_image = tf.reshape(x,[-1,28,28,1])
hidden_1 = slim.conv2d(x_image,5,[5,5])
pool_1 = slim.max_pool2d(hidden_1,[2,2])
hidden_2 = slim.conv2d(pool_1,5,[5,5])
pool_2 = slim.max_pool2d(hidden_2,[2,2])
hidden_3 = slim.conv2d(pool_2,20,[5,5])
hidden_3 = slim.dropout(hidden_3,keep_prob)
out_y = slim.fully_connected(slim.flatten(hidden_3),10,activation_fn=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y))
correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
batchSize = 50

for i in range(1001):
    batch = mnist.train.next_batch(batchSize)
    sess.run(train_step, feed_dict={x:batch[0],true_y:batch[1], keep_prob:0.5})
    if i % 100 == 0 and i != 0:
        trainAccuracy = sess.run(accuracy, feed_dict={x:batch[0],true_y:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))

testAccuracy = sess.run(accuracy, feed_dict={x:mnist.test.images,true_y:mnist.test.labels, keep_prob:1.0})
print("test accuracy %g"%(testAccuracy))

imageToUse = mnist.test.images[0]
#plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")

getActivations(hidden_1,imageToUse,1)
getActivations(hidden_2,imageToUse,2)
getActivations(hidden_3,imageToUse,3)
'''

