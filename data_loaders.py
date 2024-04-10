import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import io
import skimage
import sys


def LoadFemur():

    return np.load("FemurScan.npy")*255 

def LoadHead():
    from skimage.transform import rescale
    images = np.zeros((160,160))
    

    head = np.load("HeadScan.npy")
    head = rescale(head,scale=(1,0.7))

    head = (((head-np.min(head))/(np.max(head)-np.min(head)))*255).astype(int)
    images[0:160,0:112] =head
    return images.astype(int)
