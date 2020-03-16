
#!/usr/bin/env python

'''
Stitching sample
================
Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

# Python 2/3 compatibility
import fish_eye
import numpy as np
import cv2 as cv

import argparse
import sys
import os
from os.path import isfile, join

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

parser = argparse.ArgumentParser(prog='stitching.py', description='Stitching sample.')
parser.add_argument('--mode',
type = int, choices = modes, default = cv.Stitcher_PANORAMA,
help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('output', default = 'result.jpg',
help = 'Resulting image. The default is `result.jpg`.')
parser.add_argument('img', nargs='+', help = 'input images')

__doc__ += '\n' + parser.format_help()

def main():
	cam1 = cv.VideoCapture(1)
	cam2 = cv.VideoCapture(2)	
	cv.namedWindow("test1")
	cv.namedWindow("test2")
	img_counter1 = 1
	img_counter2 = 1
	a=5
	ret1, frame1 = cam1.read()
	ret2, frame2 = cam2.read()
	cv.imshow("test1", frame1)
	cv.imshow("test2", frame2)
	img_name1 = r"C:/yashpd16/stitching11/stitching/test/1.png"	
	img_name2 = r"C:/yashpd16/stitching11/stitching/test/2.png"	
	k1 = cv.waitKey(1000)
	k2 = cv.waitKey(1000)	
	cv.imwrite(img_name1,frame1)
	cv.imwrite(img_name2,frame2)	
	cam1.release()
	cam2.release()
	imgs=[]
	imgs.append(frame1)
	imgs.append(frame2)
	#print(imgs)
	res =[]
	stitcher = cv.Stitcher.create(0)
	status, pano = stitcher.stitch(imgs)
	imgs=[]
	img_name3 = r"C:/yashpd16/stitching11/stitching/test/pano.jpg"	
	img_name4 = r"C:/yashpd16/stitching11/stitching/test/fish.jpg"	
	cv.imwrite(img_name3,pano)	
	#print("hieee logs")
	#print(pano)
	#cv.waitKey(0)
	#cv.destroyAllWindows()
	print(status)
	cv.imwrite(img_name4,fish_eye.fisheye_func(pano))
if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
