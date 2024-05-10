##### cornerSubPix #####
# 导入库
from __future__ import print_function
import cv2 as cv 
import numpy as np
import argparse
import random as rng



source_window = 'Subpixel'
# Trackbar to adjust maxCorners threshold (for the number of corner)
maxTrackbar = 25
# make sure the color of each operation is the same
rng.seed(12345)
# corner detect function - Shi Tomasi algorithm
def goodFeaturesToTrack_Demo(val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    # corner 的特征值大于quaility level时，被认为是有效的
    qualityLevel = 0.01
    # 两个角的最小欧式距离，如果两个角小于该值，一个点会被抑制
    minDistance = 8
    # block中比较特征值，大的blocksize会提高算法稳定性，但会让角点检测变得模糊
    blockSize = 3
    # Sobel算子的大小
    gradientSize = 3
    # 用Harris 还是Shi Tomais方法
    useHarrisDetector = False
    k = 0.04
    # Copy the source image
    copy = np.copy(src)
    detected_corners = []
    # Apply corner detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (int(corners[i,0,0]), int(corners[i,0,1])), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
    # Show what you got
    cv.namedWindow(source_window)
    cv.imshow(source_window, copy)
    # Set the needed parameters to find the refined corners
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    # Calculate the refined corner locations
    corners = cv.cornerSubPix(src_gray, corners, winSize, zeroZone, criteria)
    # Write them down
    for i in range(corners.shape[0]):
        print(" -- Refined Corner [", i, "]  (", corners[i,0,0], ",", corners[i,0,1], ")")
    return corners
# Load source image and convert it to gray
parser = argparse.ArgumentParser(description='Code for Shi-Tomasi corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='pic3.png')
# Problem here
args = parser.parse_args()
src = cv.imread('C:/02_FAU/S3/Project_Thesis/Code/Improved_Aruco_Marker/subpix_roi.jpg')
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#_, src_gray = cv.threshold(src_1, 127, 255, cv.THRESH_BINARY)
# Create a window and a trackbar
cv.namedWindow(source_window,cv.WINDOW_NORMAL)
maxCorners = 100 # initial threshold
cv.createTrackbar('Threshold: ', source_window, maxCorners, maxTrackbar, goodFeaturesToTrack_Demo)
cv.imshow(source_window, src_gray)
subpix_corners = goodFeaturesToTrack_Demo(maxCorners)
print(f'detect corners : {subpix_corners}')
print(f'shape detect corners : {np.shape(subpix_corners)}')

cv.waitKey()
cv.destroyAllWindows()

def get_subpix_corners():
    return subpix_corners