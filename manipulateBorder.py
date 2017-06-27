import numpy as np
from skimage import img_as_float

def manipulateBorderArray(data, cropSize):
	mask = int(cropSize/2)

	h,w = 6, 6
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

def main():
	cont = 0
	height = width = 6
	cropSize = 3
	img = np.empty([height, width],dtype="uint8") #row,column 

	#Save Image in Numpy Array
	for row in xrange(height):
		for column in xrange(width):
			img[row][column] = cont
			cont += 1

	finalData = manipulateBorderArray(img, cropSize)
	print(finalData.shape)

if __name__ == "__main__":
    	main()


