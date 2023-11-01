import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D, Concatenate, Input, Dropout
from tensorflow.keras.activations import * 
from tensorflow.keras.models import Sequential

from th_modules.nf1_reader import apply_mask

global H
global W

def load_data(dataset_path,split=0.2,random_state=42):
    images=sorted(glob(os.path.join(dataset_path,'images','*.jpg')))
    masks=sorted(glob(os.path.join(dataset_path,'masks','*.png')))
    test_size=int(len(images)*split)
    
    train_x,valid_x=train_test_split(images,test_size=test_size,random_state=random_state)
    train_y,valid_y=train_test_split(masks,test_size=test_size,random_state=random_state)
    train_x,test_x=train_test_split(train_x,test_size=test_size,random_state=random_state)
    train_y,test_y=train_test_split(train_y,test_size=test_size,random_state=random_state)
    
    return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)

def read_image(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_COLOR)
    x=cv2.resize(x,(W,H))
    x=x/255.0
    x=x.astype(np.float32)
    #x=np.expand_dims(x,axis=0)
    return x

def read_mask(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x=cv2.resize(x,(W,H))
    x=x/255.0
    x=x.astype(np.float32)
    X=np.expand_dims(x,axis=-1)
    x=np.expand_dims(x,axis=-1)
    return x

def augment(x,y):
    original_height, original_width = tf.shape(x)[0], tf.shape(x)[1]
    # Data augmentation: Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    # Data augmentation: Random rotation (90, 180, or 270 degrees)
    rotation_angle = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32) * 90
    x = tf.image.rot90(x, k=rotation_angle // 90)
    y = tf.image.rot90(y, k=rotation_angle // 90)
    
    # Data augmentation: Random brightness and contrast adjustment
    if tf.random.uniform(()) > 0.35:
        x = tf.image.random_brightness(x, max_delta=0.5)
        x = tf.image.random_contrast(x, lower=0.4, upper=1.1)

    # Data augmentation: Random saturation and hue adjustment
    if tf.random.uniform(()) > 0.35:
        x = tf.image.random_saturation(x, lower=0.7, upper=1.2)
        x = tf.image.random_hue(x, max_delta=0.1)

    # Data augmentation: Random Gaussian noise
    if tf.random.uniform(()) > 0.35:
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.005, dtype=tf.float32)
        x = tf.add(x, noise)

    # Conditional Cutmix augmentation
    # HORIZONTAL CUTMIX
    if tf.random.uniform(()) > 0.3:
        # Cutmix augmentation: Slice the image in half and place halves back-to-back
        cut_ratio = tf.random.uniform(shape=[], minval=0.1, maxval=0.9)
        cut_width = tf.cast(tf.round(tf.cast(original_width, tf.float32) * cut_ratio), tf.int32)
        cut_position = tf.random.uniform(shape=[], minval=0, maxval=original_width - cut_width, dtype=tf.int32)
        x1, x2 = x[:, :cut_position, :], x[:, cut_position:, :]
        y1, y2 = y[:, :cut_position, :], y[:, cut_position:, :]
        x = tf.concat([x2, x1], axis=1)
        y = tf.concat([y2, y1], axis=1)
    
    # VERTICAL CUTMIX
    if tf.random.uniform(()) > 0.3:
        cut_ratio = tf.random.uniform(shape=[], minval=0.1, maxval=0.9)
        cut_height = tf.cast(tf.round(tf.cast(original_height, tf.float32) * cut_ratio), tf.int32)
        cut_position = tf.random.uniform(shape=[], minval=0, maxval=original_height - cut_height, dtype=tf.int32)
        x1, x2 = x[:cut_position, :, :], x[cut_position:, :, :]
        y1, y2 = y[:cut_position, :, :], y[cut_position:, :, :]
        x = tf.concat([x2, x1], axis=0)
        y = tf.concat([y2, y1], axis=0)
    
    # Resize the augmented images back to the original size
    x = tf.image.resize_with_crop_or_pad(x, original_height, original_width)
    y = tf.image.resize_with_crop_or_pad(y, original_height, original_width)
    return x,y

def tf_parse(x,y):
    def _parse(x,y):
        x=read_image(x)
        y=read_mask(y)
        x,y = augment(x,y)
        return x,y
    x,y=tf.numpy_function(_parse,[x,y],[tf.float32,tf.float32])
    x.set_shape([H,W,3])
    y.set_shape([H,W,1])
    return x,y

def tf_dataset(X,Y,batch_size=1):
    dataset=tf.data.Dataset.from_tensor_slices((X,Y))
    dataset=dataset.map(tf_parse)
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(10)
    return dataset

def down_block(x, filters, use_maxpool = True):
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if use_maxpool == True:
        return  MaxPooling2D(strides= (2,2))(x), x
    else:
        return x
def up_block(x,y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis = 3)([x,y])
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding= 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x
    
def Unet(input_size = (256, 256, 3), *, classes, dropout):
    # inspired by https://github.com/Nguyendat-bit/U-net/blob/main/model.py
    # modified for purposes of this model
    
    filter = [64,128,256,512,1024]
    # encode
    inp = Input(shape = input_size)
    x, temp1 = down_block(inp, filter[0])
    x, temp2 = down_block(x, filter[1])
    x, temp3 = down_block(x, filter[2])
    x, temp4 = down_block(x, filter[3])
    x = down_block(x, filter[4], use_maxpool= False)
    # decode 
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    x = Dropout(dropout)(x)
    outp = Conv2D(classes, 1, activation= 'softmax')(x) ##sigmoid useful here?
    model = models.Model(inp, outp, name = 'unet')
    #model.summary()
    return model
if __name__ == '__main__':
    model = Unet((256,256,3), classes= 2, dropout= 0.5)
    #model.summary()


def init_unet_model(input_size, weights_path=None):
    mdl = Unet(input_size = input_size, classes = 2, dropout = 0.4);
    mdl.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(), 
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
                metrics = ['acc'], 
                run_eagerly = False);

    # if loading from previously trained checkpoint
    if weights_path != None:
        mdl.load_weights(weights_path)
    return mdl

import random
from PIL import Image, ImageFilter

def analyse_n_patches_flat(imag, mask, model, network_size, binary_threshold=0.5, n_ims_max=5, patch_size=32, patch_step=0.5):
    masked_image = apply_mask(imag, mask)
    mask_height, mask_width, _ = masked_image.shape
    network_height, network_width, _ = network_size

    step = round(patch_size * patch_step) - 1

    recon_im = np.zeros((mask_height, mask_width, 1), dtype=np.float32)
    recon_im_bin = np.zeros((mask_height, mask_width, 1), dtype=np.uint8)

    patches = []
    mask_patches = []
    preds = []

    y_coords = np.arange(0, mask_height, step)
    x_coords = np.arange(0, mask_width, step)
    x_patches, y_patches = np.meshgrid(x_coords, y_coords)
    
    x_patches = x_patches.flatten()
    y_patches = y_patches.flatten()
    
    x_coords_valid = x_patches[x_patches + patch_size < mask_width]
    y_coords_valid = y_patches[y_patches + patch_size < mask_height]

    img_patches = [masked_image[y:y+patch_size, x:x+patch_size] for y, x in zip(y_coords_valid, x_coords_valid)]
    mask_patches = [mask[y:y+patch_size, x:x+patch_size] for y, x in zip(y_coords_valid, x_coords_valid)]

    valid_indices = [i for i, mask_patch in enumerate(mask_patches) if mostly_white_px(mask_patch, thresh=0.9)]

    img_patches = np.array([cv2.resize(patch, (network_height, network_width), interpolation=cv2.INTER_LINEAR) for patch in img_patches])
    batch_preds = model.predict(img_patches)[:, :, :, 1]

    for i in valid_indices:
        x_batch, y_batch = x_coords_valid[i], y_coords_valid[i]

        pred = batch_preds[i]
        pred = pred.reshape((network_height, network_width, 1))
        pred = cv2.resize(pred, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        pred = np.expand_dims(pred, axis=2)
        
        preds.append(pred)
        patches.append(img_patches[i])
        mask_patches.append(mask_patches[i])

        binary_pred = imbinarise(pred)
        recon_im[y_batch:y_batch+patch_size, x_batch:x_batch+patch_size] += pred
        recon_im_bin[y_batch:y_batch+patch_size, x_batch:x_batch+patch_size] += binary_pred

        if n_ims_max != -1 and len(preds) >= n_ims_max:
            break

    return patches, mask_patches, preds, apply_mask(recon_im, mask), apply_mask(recon_im_bin, mask)

#def normalise_reconstruction(recon_im,mask):
#    recon_im = apply_mask(recon_im,mask)
#    return recon_im

def count_white_pixels(mask_patch):
    patch_size = (mask_patch.shape)[0]
    n_black_pixels = (mask_patch <= (0.1 * mask_patch.max())).sum()
    n_white_pixels = (patch_size**2) - n_black_pixels
    return n_white_pixels

def mostly_white_px(mask_patch,thresh = 0.95):
    
    patch_size = (mask_patch.shape)[0]
    min_white_pixels = (patch_size**2) * thresh
    n_white_pixels = count_white_pixels(mask_patch)
    #print(1.0 * n_white_pixels / min_white_pixels)
    return (n_white_pixels > min_white_pixels)

def imbinarise(image,thresh=0.8):
    #thresh_image = cv2.adaptiveThreshold(image.astype(np.uint8)*255, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    thresh_image = (image > 0.8).astype(np.uint8)
    #print(thresh_image.shape)
    return thresh_image

def morph_open_image(image,diameter=5):
    
    #kernel_size = (3, 3)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    major_axis = diameter
    minor_axis = diameter
    angle = 45  # Angle in degrees, rotation of the oval
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (major_axis, minor_axis))

    if angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D((major_axis // 2, minor_axis // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (major_axis, minor_axis), flags=cv2.INTER_LINEAR)


    # Perform morphological opening
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Convert to uint8
    opened_image = opened_image.astype(np.uint8)
    return opened_image

