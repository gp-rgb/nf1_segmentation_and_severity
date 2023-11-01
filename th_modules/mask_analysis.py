import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from th_modules.nf1_reader import apply_mask
from th_modules.unet_segmenter import count_white_pixels

def tumour_density(skin_mask,tumour_mask):
    if skin_mask.shape != tumour_mask.shape:
        print("incompatible shapes")
        return 0
    
    sk_n = count_white_pixels(skin_mask)*1.0
    tm_n = count_white_pixels(tumour_mask)*1.0

    percentage = tm_n/sk_n
    return percentage


def cluster_stats(tumour_mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tumour_mask, connectivity=8)
    n_clusters = num_labels - 1
    
    areas = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        areas.append(area)

    return n_clusters, areas

def areas_to_hist(areas,n_bins=10,n_clusters = None,perc=None):
    fig = plt.figure()
    plt.hist(areas, bins=n_bins, edgecolor='black', alpha=0.7)  # You can adjust the number of bins as needed

    # Add labels and a title
    plt.xlabel('Tumour Cluster Size (Pixels)')
    plt.ylabel('Frequency')
    if (n_clusters is not None) and (perc is not None):
        plt.title(f'Cluster Size Distribution, N={n_clusters}, {perc:.4f}% of Skin')
    else:
        plt.title('Distribution of Tumour Cluster Sizes')

    # Show the histogram
    fig.show()