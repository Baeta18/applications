
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
def generateFusion(groundPath,outputPath,instance,fusionInstances,data,crops,probsUsed):
	#print("Generate fusion")
	imageCont = 0
	
	tam= int(math.sqrt(data[0].shape[0]))
	resultPath = outputPath + str(instance) + "_instance/"

	if os.path.exists(resultPath) != True:
		#print("Creating folder: " + resultPath)
                os.makedirs(resultPath)

	
	#print("Opening file")
	#result = open(resultPath + "result.txt",'a')
	#print(crops)
	#print(len(data))	
	count = 0



	for i in xrange(0,len(data)):
		#print(imageCont)
		#print("Predicting image " + fusionInstances[imageCont] )
		groundFile = groundPath + fusionInstances[imageCont] + "/mascara.pgm"
		#print("Prob used " + probsUsed[i] )

		#print("Loading ground " + groundFile)
		#fusion_map = np.zeros((tam,tam))
		mask = read_image_P2_int16(groundFile)
		#predicts = np.zeros(1000000)
		#grounds = np.zeros(1000000)
		#acc = 0.0


		if(i%len(fusionInstances) == 0):
			rightsGlobal = {}
			rightsGlobal[0] = 0.0
			rightsGlobal[1] = 0.0

		
		grounds = {}
		rights = {}
		grounds[0] = 0.0
		rights[0] = 0.0
		grounds[1] = 0.0
		rights[1] = 0.0
		

		for j in xrange(1000000):
			posY = int(data[i][j][0])
			posX = int(data[i][j][1])
			ground = mask[posY][posX] if mask[posY][posX] == 0 else 1
			grounds[ground] += 1

			if(data[i][j][2] > data[i][j][3]):
				prediction = 0
			else:
				prediction = 1


			if(ground == prediction):
				rights[prediction] += 1

		print(str(rights[0]/grounds[0]*100) + ";" + str(rights[1]/grounds[1]*100))
		count += 1

		if(count != len(fusionInstances)):
			rightsGlobal[0] += (rights[0]/grounds[0]*100)
			rightsGlobal[1] += (rights[1]/grounds[1]*100)
			imageCont += 1
		else:
			rightsGlobal[0] += (rights[0]/grounds[0]*100)
			rightsGlobal[1] += (rights[1]/grounds[1]*100)
			imageCont = 0
			count = 0

			print("Mean " + probsUsed[i] )
			#print(str(rightsGlobal[0]/3) + ";" + str(rightsGlobal[1]/3) + "\n")
			print("\n")


	#print("Global")
	#print("Class 0 " + str(rightsGlobal[0]/3) + "%")
	#print("Class 1 " + str(rightsGlobal[1]/3) + "%")

		#cur_kappa = cohen_kappa_score(grounds, predicts)
		#print("Acc/Kappa " + str(acc/len(grounds)) + ";" + str(cur_kappa))
		#result.write(str(instance) + "_" + fusionInstances[imageCont] + ": Kappa: " + str(cur_kappa) + " Acc: " + str(acc/len(grounds)*100) + "\n")
		#fusionFile =  resultPath + str(instance) + "_fusion_" + fusionInstances[imageCont] + ".png"
		#print("Saving image '" + fusionFile + "'")
		#scipy.misc.imsave(fusionFile , fusion_map)

	#result.close()



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
	
	arc3 = [17,25,33,41]
	arc4 = [33,41,49]
	arc5 = [49,57]
	arc6 = [57]
	probs = 0
	probUsed = []
	for cropSize in cropSizes: 
		for i in fusionInstances:
			if(int(cropSize) in arc3):
				probFile = dataPath + "fused/" + str(instance) + "_probs_3_blocks_" + str(cropSize) + "_final_" + i + ".npy" 			
				#print("Loading file: " + probFile)
				probsData.append(np.load(probFile))
				probUsed.append(probFile)
				probs += 1
	for cropSize in cropSizes: 
		for i in fusionInstances:
			if(int(cropSize) in arc4):		
				probFile = dataPath + "fused/" + str(instance) + "_probs_4_blocks_" + str(cropSize) + "_final_" + i + ".npy" 			
				#print("Loading file: " + probFile)
				probsData.append(np.load(probFile))
				probUsed.append(probFile)
				probs += 1

	for cropSize in cropSizes: 
		for i in fusionInstances:
			if(int(cropSize) in arc5):		
				probFile = dataPath + "fused/" + str(instance) + "_probs_5_blocks_" + str(cropSize) + "_final_" + i + ".npy" 			
				#print("Loading file: " + probFile)
				probsData.append(np.load(probFile))
				probUsed.append(probFile)
				probs += 1

	for cropSize in cropSizes: 
		for i in fusionInstances:
			if(int(cropSize) in arc6):		
				probFile = dataPath + "fused/" + str(instance) + "_probs_6_blocks_" + str(cropSize) + "_final_" + i + ".npy" 			
				#print("Loading file: " + probFile)
				probsData.append(np.load(probFile))
				probUsed.append(probFile)
				probs += 1


	probs = (probs/len(fusionInstances))
	generateFusion(groundPath,outputPath,instance,fusionInstances,probsData,probs,probUsed)
	


if __name__ == "__main__":
	main()
