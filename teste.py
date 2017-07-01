
import os
import os.path
import gc
import sys
import random
import scipy.misc
import numpy as np
import subprocess
import math


from PIL import Image, ImageOps

tam = 1000
a = np.zeros((tam*tam,4))
b = np.zeros((tam*tam,4))
c = np.zeros((tam*tam,4))

fusion_map = np.zeros((tam,tam))

for i in xrange(tam*tam):

	if(i < 500000):
		a[i][2] = 0.9
		a[i][3] = 0.1
		b[i][2] = 0.9
		b[i][3] = 0.1
		c[i][2] = 0.9
		c[i][3] = 0.1
	else:
		a[i][3] = 0.9
		a[i][2] = 0.1
		b[i][3] = 0.9
		b[i][2] = 0.1
		c[i][3] = 0.9
		c[i][2] = 0.1

for i in xrange(tam*tam):
	posX = a[i][0]
	posY = a[i][0]
	class0 = a[i][2] + b[i][2] + c[i][2]
	class1 = a[i][3] + b[i][3] + c[i][3]
	
	if class0 > class1:
		prediction = 0
	else:
		prediction = 1

	fusion_map[posY][posX] = prediction
	
	scipy.misc.imsave("teste.png", fusion_map)
