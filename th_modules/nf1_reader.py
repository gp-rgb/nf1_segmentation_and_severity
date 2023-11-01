from matplotlib.image import imsave
import numpy as np # linear algebra
import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PIL import Image
import os
from glob import glob
from sklearn.model_selection import train_test_split
import random

def shuffle_two_lists(l1,l2):
    # Combine the two lists into pairs using zip
    combined = list(zip(l1, l2))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the shuffled list back into separate lists
    shuf1, shuf2 = zip(*combined)
    return shuf1,shuf2

def get_nf1_paths(root_dir,n=5,demo=True):
    #demo=True for non-confidential images
    imags=[]
        
    for root, _, fns in os.walk(root_dir):
        for filename in fns:
            if demo and (filename[0]!='d'):
                continue
            if filename.endswith('.jpeg'):
                p = os.path.join(root,filename)
                #imag = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.uint8())
                imags.append(p)
  
            if (len(imags)>=n) and (n != -1):
                random.shuffle(imags)
                return imags
        
    random.shuffle(imags)
    return imags
    
def read_nf1_paths(im_paths):
    imags = []
    
    for i in im_paths:
        imag = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB).astype(np.uint8())
        imags.append(imag)

    return imags


def crop_resize_nf1(imags,siz=1024):
    proc_imags=[]
    for imag in imags:

        imag = tf.image.resize(imag, [siz, siz])
        
        proc_imags.append(imag.numpy())
    return proc_imags

def equalise_nf1(imag,channels=[0,1,2]):
    for channel in channels:
        imag[:,:,channel] = cv2.equalizeHist(imag[:,:,channel])
    return imag

def to_grayscale(imag,invert=False):
    gray_image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    if invert:
        # Invert the grayscale image
        inverted_image = 255 - gray_image
        return inverted_image
    return gray_image

def blur_image(image,kernel=5):
    # Apply Gaussian blur
    kernel_size = (kernel, kernel)  # You can adjust the kernel size to control the blur effect
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def apply_mask(imag,mask):
    masked_image = cv2.bitwise_and(imag, imag, mask=mask) 
    norm_maskim = masked_image 
    return norm_maskim


def draw_bounding_boxes(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle


def mask_to_bounding_boxes(image,mask,title = None):
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_boxes = mask.copy()

    draw_bounding_boxes(image_with_boxes, contours)

    fig = plt.figure()
   
    plt.imshow(image_with_boxes)
    if title is not None:
        plt.title(title)
    else:
        plt.title('Image with Bounding Boxes')

    fig.show()

def get_nf1_images(root_dir,n_images, siz = 1024, n_paths=-1, demo=True):
    nf1_im_paths = get_nf1_paths(
        root_dir,
        n=n_paths,
        demo=demo,
        )

    nf1_im_paths = nf1_im_paths[0:n_images]

    nf1_rawims = read_nf1_paths(nf1_im_paths)

    nf1_images = crop_resize_nf1(nf1_rawims, siz=siz)
    return nf1_images
