import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import io
import skimage
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import sys
from mw_funcs_ND import *
from data_loaders import *

image = LoadFemur() # Load in the image
ViewDataset(image) # View the ultrasound scans and intensities

# -- Look at zoom region ------
w = 30 # EDIT ME
h = 50 # EDIT ME

ViewZoom(image,w,h,L=10)


Ti = 130 # EDIT ME
Tf = 200 # EDIT ME

# -- View the threshold image ------
threshold_image=BandwidthThresholdImage(image, Ti,Tf) # threshold the image
BandWidthViewThreshold(image,threshold_image,Ti,Tf) # view the thresholded the image
 #Line Detection --:
l_min = 52 # EDIT ME

lines = HoughLines(threshold_image, l_min) # Generate the Hough lines
ShowHoughLines(threshold_image,image,lines,ext=1) # View the Hough lines
ComputeLength(lines) # Calculate the length of the lines
lines = lines[0]
print(lines)
linelengthx = abs(lines[0][0]-lines[1][0])
print(linelengthx)
linelengthy = abs(lines[0][1]-lines[1][1])
linelength = np.sqrt((linelengthx**2)+(linelengthy**2))
print(linelength)
Length_pixels = linelength #EDIT HERE
Conversion_pixel2mm = 0.6 #1 pixel = 5mm

Length_mm =Length_pixels*Conversion_pixel2mm # multiply together
print(f"Length of stand = {Length_mm:.1f} mm")

femur_length_mm= Length_mm# EDIT ME
FindFetalAge(femur_length_mm)


image = LoadHead() # Load in the image
ViewDataset(image) # View the ultrasound scans and intensities

# -- Look at zoom region ------
w = 10# EDIT ME
h = 10 # EDIT ME

ViewZoom(image,w,h,L=5)
Ti = 10 #Edit me
Tf = 100 #Edit me

edges = canny(image, sigma=3, low_threshold=Ti, high_threshold=Tf)

# Detect two radii for size r=20, r =35 (e.g. search the image for these radii)
r_1 = 40
r_2 = 45
hough_radii = np.arange(r_1,r_2, 2)
hough_res = hough_circle(edges, hough_radii)

num_circs = 3 # How many circles we wants
red_radii = showcircles(hough_res, hough_radii, image,num_circs=num_circs)
max_radius = np.max(red_radii)
print("Maximum radius: ", max_radius, "pixels")
pi = np.pi



head_circum_mm= max_radius *1.67*2# ADD VALUE HERE
FindFetalAge_head(head_circum_mm)