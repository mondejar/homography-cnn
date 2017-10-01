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

fileList = open(trainFilename,'r') 


#patch_I2 = cv2.cvtColor(patch_I2, cv2.COLOR_GRAY2RGB)
#cv2.polylines(patch_I2,  np.int32([gtCorner]), 1, (255, 25, 255), 1)
