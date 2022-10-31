#%% 
from itertools import product
from unittest import result
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m
import time as t
from statistics import *



def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
	
	
def cal_lum(r,g,b):
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        lum = int(m.sqrt(lum/256)*256)     
    
        return lum
		
def gray_lev(im):

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            r,g,b = im[i,j]
        
            im[i,j] = 3*[cal_lum(r,g,b)]

    return im
        	
def contrast(gray_im):
    shape = gray_im.shape
    try:
        table = np.array([gray_im[i,j][0]/255 for i in range(shape[0]) for j in range(shape[1])])
    except:
        table = np.array([gray_im[i,j]/255 for i in range(shape[0]) for j in range(shape[1])])

    gray_avg = sum(table)/len(table)
    
    gray_var = sum([(table[i]-gray_avg)**2 for i in range(len(table))])
    
    var_max = len(table) * (255**2)/12
    
    return gray_avg , gray_var/var_max # <= 1
    
def entropy(gray_im):
    t1 = t.time()
    shape = gray_im.shape
    table = np.zeros(256)
    for i in range(shape[0]):
        for j in range(shape[1]):
            try:
                gray_nb = gray_im[i,j][0] 
            except:
                gray_nb = gray_im[i,j]               
            
            table[gray_nb] += 1
    print("entropy time =", t.time() - t1 )
    return table
    
    

def change(pixel_color,avg,var):
    new_pixel_color = (pixel_color/255 - avg)*3 + 1/2
    
    new_pixel = int(new_pixel_color*255)
    
    return new_pixel

def separate_grey(im,avg,var):
    
    for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                try:
                    r,g,b = im[i,j]
                except:
                    r = im[i,j]
                pixel_color = r
                
                im[i,j] = change(pixel_color,avg,var)
        
    return im
    
    
    #%% Import image
if False:    
    image = cv2.imread("Assets\img test 2.jpg")
    
    cv2.imshow('Initial image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    #%% Grey image 1
    
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg,var = contrast(gray_im)
    
    equ = cv2.equalizeHist(gray_im)
    res = np.hstack((gray_im,equ)) #stacking images side-by-side
    cv2.imshow('Equalized histogramm',res)
    
    print("Contrast:")
    print("AVG",avg,"VAR",var)
    
    cv2.imshow('Grey image 1',gray_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    table = entropy(gray_im)
    plt.plot([i for i in range(len(table))],[table[i] for i in range(len(table))])
    plt.title('Entropy graph for image 1')
    
    plt.show()
    
    
    
    
    #%%Grey image 2
    
    gamma_value = 5
    
    image1 = adjust_gamma(image,gamma_value)
    
    gray_im = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    avg,var = contrast(gray_im)
    print("Contrast:")
    print("AVG",avg,"VAR",var)
    
    cv2.imshow('image 2',gray_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    table = entropy(gray_im)
    plt.plot([i for i in range(len(table))],[table[i] for i in range(len(table))])
    plt.title('Entropy graph for image 2')
    plt.show()
    
    
    
    #%% Grey image 3
    
    gray_im0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    avg,var = contrast(gray_im0)
    gray_im = separate_grey(gray_im0,avg,var)
    
    avg,var = contrast(gray_im)
    
    print("Contrast:")
    print("AVG",avg,"VAR",var)
    
    cv2.imshow('image 3',gray_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    table = entropy(gray_im)
    plt.plot([i for i in range(len(table))],[table[i] for i in range(len(table))])
    plt.title('Entropy graph for image 3')
    plt.show()
    
