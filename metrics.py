import os
import cv2
import tensorflow as tf
import numpy as np

def normalize(input_data):
    return (np.float32(input_data) - 127.5)/127.5 

def denormalize(input_data):
    return np.uint8(np.int32((np.float32(input_data) * 127.5) + 127.5 ))

# Input is array of images of size 100x100
def bicubic(array):
    for i in range(len(array)):
        array[i] = cv2.resize(array[i], dsize=(400, 400), interpolation=cv2.INTER_CUBIC)  
    return array

# Input is array of images of size 100x100
def bilinear(array):
    for i in range(len(array)):
        array[i] = cv2.resize(array[i], dsize=(400, 400), interpolation=cv2.INTER_LINEAR)  
    return array

# Input is array of images of size 100x100
def nearest(array):
    for i in range(len(array)):
        array[i] = cv2.resize(array[i], dsize=(400, 400), interpolation=cv2.INTER_NEAREST)  
    return array

# Input is array of reconstructed images and ground truth images of size 400x400
def ssim_psnr(super_images,high_images):
    tf.reset_default_graph()
    sr = tf.placeholder(tf.float32,(None,400,400,3))
    hr = tf.placeholder(tf.float32,(None,400,400,3))
    ssim = tf.image.ssim(sr,hr,255)
    psnr = tf.image.psnr(sr,hr,255)
    sess = tf.Session()
    ssim,psnr = sess.run([ssim,psnr],feed_dict={sr:super_images,hr:high_images})
    ssim = np.mean(ssim)
    psnr = np.mean(psnr)
    return (ssim,psnr)
