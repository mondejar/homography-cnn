#!/usr/bin/python
################################################################################
#
# Author: V. Mondejar-Guerra
#
# Create at 30 Sep 2017
# Last modification: 1 Oct 2017
################################################################################

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
import random
import sys


""" 
This code prepares the data for train the model.
Given a directory with the images this code creates for each image:
	- I1 a patch of size NxN
	- I2 a patch of size NxN that is a perspective transformation of I1
	- The 4 coordinates in I2 that enclose I1

	    		

python dataset_path max_number_images number_transformations patch_size
				
Example call:

python create_dataset.py ../mirflickr/ 1000 5 128

"""

# Params
outImPath = 'data/train_data/'		   # Dir in which the new patches are saved
trainFilename = 'data/train_data_list.txt' # Full file path referencing the patches and ground truth
verbose = False # set True to display the process

# Generate a random affine transform over the four corners
def affine_transform( patchSize, img, width, height, randomDispFactor ):
	
	w = int(round(randomDispFactor * patchSize))
	h = int(round(randomDispFactor * patchSize))
	corners = np.float32([[w, h], [w, patchSize - h], [patchSize - w, h], [patchSize - w, patchSize - h]])
	cornersT = np.float32([[1, 1], [1, 1], [1, 1], [1, 1]])

	for p in range(0,4):
		randValue = random.uniform(1, 100) / 100  
		x =  int(round(((w*2 * randValue)))  + corners[p][0] )
		randValue = random.uniform(1, 100) / 100
		y = int(round(((h*2 * randValue))) + corners[p][1]	)
		cornersT[p] = np.float32([x, y])

	persT = cv2.getPerspectiveTransform(corners, cornersT)
	warpImg = cv2.warpPerspective(img, persT, (width, height))
	#print 'Perspective matrix ', persT

	return warpImg, persT


def main(argv):

	if len(argv) < 5:
		print 'Error, incorrect number of args:\n python dataset_path max_number_images number_transformations patch_size\n'
		sys.exit(1)

	if not os.path.exists(outImPath):
		os.mkdir(outImPath)

	imDatasetPath = argv[1]  # Dir that contains the original images
	maxImg = int(argv[2])
	numWarps = int(argv[3]) # number of warps per image
	patchSize =  int(argv[4]) # patch size 

	# create file
	fileList = open(trainFilename,'w') 
	imFiles = [f for f in listdir(imDatasetPath) if isfile(join(imDatasetPath, f))]

	numIm = 0
	for imFile in imFiles:
		if numIm >= maxImg:
			sys.exit(1)
		# Read image
		img = cv2.imread(imDatasetPath + imFile, 0)
		if not img is None:

			# for each 
			height, width = img.shape[:2]
			if height > patchSize * 3 and width > patchSize * 3:
				print str(numIm), ' Load: ', imDatasetPath + imFile
				x1 = (width/2)- (patchSize/2)
				x2 = (width/2) + patchSize/2
				y1 = (height/2)- (patchSize/2)
				y2 = (height/2) + patchSize/2

				patchLocation = np.float32( [ [x1, y1], [x2, y1], [x2, y2], [x1, y2]])
				patchLocation = np.array([patchLocation])
				patch_I1 = img[y1:y2, x1:x2] # Crop from x, y, w, h -> 100, 200, 300, 400
				namePatchI1 = outImPath + str(numIm) + '_I1' + '.png'
				cv2.imwrite( namePatchI1, patch_I1)	

				for nW in range(0, numWarps):
					# Perform several transformations for the same image, 
					# in order to make the network learn different corner outputs
					# from the same patch I1

					# Extract a patch in the center of size 128x128
					# Perform the 4-corner perspective change!
					warpImg, Hom = affine_transform(patchSize, img, width, height, 0.05)		
					patchHom = cv2.perspectiveTransform(patchLocation, Hom)

					# Generate some extra offset on the I2 coordinates
					# This factor will make the extra margin on I2 bigger
					randomDispFactor = 0.7
					x1_off = int(round(randomDispFactor * patchSize))  * (random.uniform(1, 100) / 100)
					x2_off = int(round(randomDispFactor * patchSize))  * (random.uniform(1, 100) / 100)
					y1_off = x1_off
					y2_off = x2_off
					#y1_off = int(round(randomDispFactor * patchSize))  * (random.uniform(1, 100) / 100)
					#y2_off = int(round(randomDispFactor * patchSize))  * (random.uniform(1, 100) / 100)

					patchHom[0][0][0] = patchHom[0][0][0] - x1_off
					patchHom[0][0][1] = patchHom[0][0][1] - y1_off
					patchHom[0][1][0] = patchHom[0][1][0] + x2_off
					patchHom[0][1][1] = patchHom[0][1][1] - y1_off
					patchHom[0][2][0] = patchHom[0][2][0] + x2_off
					patchHom[0][2][1] = patchHom[0][2][1] + y2_off
					patchHom[0][3][0] = patchHom[0][3][0] - x1_off
					patchHom[0][3][1] = patchHom[0][3][1] + y2_off
					
					xh1, yh1 = patchHom[0].min(0)
					xh2, yh2 = patchHom[0].max(0)
					xh1 = int(round(xh1))
					xh2 = int(round(xh2))
					yh1 = int(round(yh1))
					yh2 = int(round(yh2))

					#print '\n Y[', yh1, ':', yh2, ']  X [', xh1, ':', xh2, ']'
					patch_I2 = warpImg[yh1:yh2, xh1:xh2]
					wh = xh2 - xh1
					hh = yh2 - yh1

					gtCorner = np.float32([[1, 1], [1, 1], [1, 1], [1, 1]])
					gtCorner[0] = np.float32([(patchHom[0][0][0] - xh1) + x1_off     , (patchHom[0][0][1] - yh1) + y1_off])
					gtCorner[1] = np.float32([wh - (xh2 - patchHom[0][1][0]) - x2_off, (patchHom[0][1][1] - yh1) + y1_off])
					gtCorner[2] = np.float32([wh - (xh2 - patchHom[0][2][0]) - x2_off, hh - (yh2 - patchHom[0][2][1]) - y2_off])
					gtCorner[3] = np.float32([(patchHom[0][3][0] - xh1) + x1_off     , hh - (yh2 - patchHom[0][3][1]) - y2_off])

					fs_x = float(patchSize) / float(wh)
					fs_y = float(patchSize) / float(hh)
					for p in range(0, 4):
						gtCorner[p][0] = int(round(gtCorner[p][0] * fs_x))
						gtCorner[p][1] = int(round(gtCorner[p][1] * fs_y))

					# Scale patch_I2 to patchSize and adjust the gtCorners!
					patch_I2 = cv2.resize(patch_I2, (patchSize, patchSize))

					# TODO Apply more changes:
					# - Blurring
					# - Illumination 
					# - Occlusion??
					# - ...
					
					# Export patch warp
					namePatchI2 = outImPath + str(numIm) + '_I' + str(nW + 2) + '.png'
					cv2.imwrite( namePatchI2, patch_I2)	

					# add line to file
					fileList.write( namePatchI1 + ' ' + namePatchI2)
					gtCorner = gtCorner / patchSize
					for p in range(0, 4):
						
						fileList.write(' ' + str(gtCorner[p][0]) + ' ' + str(gtCorner[p][1]))
					fileList.write('\n')

					if verbose:
						cv2.polylines(img,  np.int32(patchLocation), 1, 255, 7)
						cv2.polylines(warpImg,  np.int32(patchHom), 1, 0, 4)
						#cv2.imshow('full image', img)
						#cv2.imshow('full image warped', warpImg)

						for p in range(0, 4):	
							print p, ': ', gtCorner[p] 
							cv2.circle(patch_I2, (int(round(gtCorner[p][0] * patchSize)), int(round(gtCorner[p][1] * patchSize))), 5, 255, 1)

						cv2.imshow('I1', patch_I1)
						cv2.imshow('I2', patch_I2)
						cv2.waitKey(0)
						cv2.destroyAllWindows()

				numIm = numIm + 1

	fileList.close()



if __name__ == "__main__":
    main(sys.argv)
