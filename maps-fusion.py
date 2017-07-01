
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


#x,y,prob-class0,prob-class1
def generateFusion(outputPath,instance,fusionInstances,data,crops):
	print("Generate fusion")
	imageCont = 0

	tam= int(math.sqrt(data[0].shape[0]))
	

	for i in xrange(0,len(data),crops):
		print("Predicting image " + fusionInstances[imageCont])
		fusion_map = np.zeros((tam,tam))


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

			fusion_map[posY][posX] = prediction

		fusionFile =  outputPath + str(instance) + "_fusion_" + fusionInstances[imageCont] + ".png"
		print("Saving image '" + fusionFile + "'")
		scipy.misc.imsave(fusionFile , fusion_map)

		imageCont += 1



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
	#dataPath = sys.argv[index]
	dataPath = ""

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
			probFile = dataPath + str(instance) + "_probs_" + str(cropSize) + "_final_" + i + ".npy" 
			print("Loading file: " + probFile)
			probsData.append(np.load(probFile))

	generateFusion(instance,outputPath,fusionInstances,probsData,3)



if __name__ == "__main__":
	main()