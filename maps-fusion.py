
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
import pylab as Plot

from PIL import Image, ImageOps
from os import listdir
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_similarity_score
from skimage import img_as_float



def printParams(listParams):
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	for i in xrange(1,len(sys.argv)):
		print listParams[i-1] + '= ' + sys.argv[i]
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	

def main():


	listParams = ['dataPath', 'outputPath(for model, images, etc)', 'instance', 'fusion-instances','cropSize','fusion-type']
	printParams(listParams)

	#training images path
	index = 1
	dataPath = sys.argv[index]

	#output path
	index = index + 1
	outputPath = sys.argv[index]

	#output path
	index = index + 1
	instance = sys.argv[index]

	#fusin instances
	index = index + 1
	fusionInstances = sys.argv[index].split(',')

	#cropsize
	index = index + 1
	cropSize = int(sys.argv[index])

	#cropsize
	index = index + 1
	fusion_type = int(sys.argv[index])

	probsData = []

	for instance in fusionInstances:
		print(instance)



if __name__ == "__main__":
	main()