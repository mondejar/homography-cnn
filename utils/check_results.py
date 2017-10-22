#!/usr/bin/python
################################################################################
#
# Author: V. Mondejar-Guerra
#
# Create at 1 Oct 2017
# Last modification: 1 Oct 2017
################################################################################

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
import random

"""
Given a file with the structure:
	pathPatchI1 pathPatchI2 x1 y1 x2 y2 x3 y3 x4 y4
	..
	..

Load the two patches and warp the 4 corners from patchI2 
"""

filename = '../data/train_data_list.txt'

fileList = open(filename,'r') 
lines = fileList.readlines() 

patchSize = 128
corners = np.float32([[0, 0], [patchSize, 0], [patchSize, patchSize], [0, patchSize]])
corners_warp = np.float32([[1, 1], [1, 1], [1, 1], [1, 1]])

for l in lines:
	ls = l.split(' ', 10)
	filenameI1 = ls[0] 
	filenameI2 = ls[1]
	for p in range(0, 4):
		corners_warp[p][0] = float(ls[ (p*2) + 2])
		corners_warp[p][1] = float(ls[ (p*2) + 3])
	
	I1 = cv2.imread('../' + filenameI1, 0)
	I2 = cv2.imread('../' + filenameI2, 0)

	persT = cv2.getPerspectiveTransform(corners_warp, corners)
	I2unwarp = cv2.warpPerspective(I2, persT, (patchSize, patchSize))

	cv2.imshow('I1', I1)
	cv2.imshow('I2', I2)
	cv2.imshow('I2 unwarp', I2unwarp)

	cv2.waitKey(0)


#patch_I2 = cv2.cvtColor(patch_I2, cv2.COLOR_GRAY2RGB)
#cv2.polylines(patch_I2,  np.int32([gtCorner]), 1, (255, 25, 255), 1)
