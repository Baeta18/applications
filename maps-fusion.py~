
import os
import os.path
import gc
import sys
import scipy.misc
import numpy as np
import math
import colorsys


from PIL import Image, ImageOps
from os import listdir
from sklearn.metrics import cohen_kappa_score

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


#x,y,prob-class0,prob-class1
def generateFusion(groundPath,outputPath,instance,fusionInstances,data,crops):
	print("Generate fusion")
	imageCont = 0
	
	tam= int(math.sqrt(data[0].shape[0]))
	print(outputPath)
	'''
	print("Opening file")
	result = open(outputPath + "result.txt",'a')

	for i in xrange(0,len(data),crops):
		print("Predicting image " + fusionInstances[imageCont])
		groundFile = groundPath + fusionInstances[imageCont] + "/mascara.pgm"
		print("Loading ground " + groundFile)
		fusion_map = np.zeros((tam,tam))
		mask = read_image_P2_int16(groundFile)
		predicts = np.zeros(1000000)
		grounds = np.zeros(1000000)

		for j in xrange(1000000):
			posY = int(data[i][j][0])
			posX = int(data[i][j][1])
			class0 = 0
			class1 = 0

			for k in xrange(crops):
				class0 += data[i+k][j][2] 
				class1 += data[i+k][j][3] 

			if(class0 > class1):
				prediction = 0
			else:
				prediction = 1

			grounds[j] = mask[posY][posX] if mask[posY][posX] == 0 else 1
			predicts[j] = prediction			
			fusion_map[posY][posX] = prediction

		cur_kappa = cohen_kappa_score(grounds, predicts)
		print("Kappa " + str(cur_kappa))
		result.write(str(instance) + "_" + fusionInstances[imageCont] + ": " + str(cur_kappa) + "\n")
		fusionFile =  outputPath + str(instance) + "_fusion_" + fusionInstances[imageCont] + ".png"
		print("Saving image '" + fusionFile + "'")
		scipy.misc.imsave(fusionFile , fusion_map)

		imageCont += 1
	result.close()
	'''


def printParams(listParams):
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	for i in xrange(1,len(sys.argv)):
		print listParams[i-1] + '= ' + sys.argv[i]
	print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
	

def main():


	listParams = ['groundPath', 'dataPath', 'outputPath(for model, images, etc)', 'instance', 'fusion-instances','cropSize','fusion-type']
	printParams(listParams)

	#training images path
	index = 1
	groundPath = sys.argv[index]

	#training images path
	index = index + 1
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
	cropSizes = sys.argv[index].split(',')

	#cropsize
	index = index + 1
	fusion_type = int(sys.argv[index])

	probsData = []
	

	for i in fusionInstances:
		for cropSize in cropSizes: 
			print(cropSize)
			if(cropSize == 25):
				probFile = dataPath + str(instance) + "_probs_" + str(cropSize) + "_final_" + i + ".npy" 
			else:
				probFile = dataPath + str(instance) + "_probs_4_blocks_" + str(cropSize) + "_final_" + i + ".npy" 			
			print("Loading file: " + probFile)
			probsData.append(np.load(probFile))

	generateFusion(groundPath,outputPath,instance,fusionInstances,probsData,3)



if __name__ == "__main__":
	main()
