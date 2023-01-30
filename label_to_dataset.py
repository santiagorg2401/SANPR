import os
import numpy as np
from PIL import Image 
import random

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from HelperFunctions import ImportHelperFunctions as IHF
from HelperFunctions import ModelHelperFunctions as MHF
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory


class LicencePlateColorGenerator(layers.Layer):
    def __init__(self, **kwargs):
        super(LicencePlateColorGenerator, self).__init__(**kwargs)

    def call(self, images, training=None):
        if not training:
            return images

        yellow_plate = tf.image.adjust_contrast(tf.math.multiply(images, -1), 1.4)
        white_plate = tf.image.adjust_saturation(yellow_plate, 0)

        return [images, yellow_plate, white_plate]


def save_image_and_label(lic_image):
    global image_index
    #Saving the image with a new name
    if image_index <= train_split:
        lic_image.save("train/images/ccpd_" + str(image_index) + ".jpg")
        #Saving txt files for labels
        with open("train/labels/ccpd_" + str(image_index) + ".txt", 'w') as f:
            f.write("0 " + str(norm_top_left[0,0]) + " " + str(norm_top_left[0,1])+ " " + str(norm_bot_right[0,0]) + " " + str(norm_bot_right[0,1]))
    if image_index > train_split and image_index <= valid_split:
        lic_image.save("valid/images/ccpd_" + str(image_index) + ".jpg")
        #Saving txt files for labels
        with open("valid/labels/ccpd_" + str(image_index) + ".txt", 'w') as f:
            f.write("0 " + str(norm_top_left[0,0]) + " " + str(norm_top_left[0,1])+ " " + str(norm_bot_right[0,0]) + " " + str(norm_bot_right[0,1]))
    if image_index > valid_split:
        lic_image.save("test/images/ccpd_" + str(image_index) + ".jpg")
        #Saving txt files for labels
        with open("test/labels/ccpd_" + str(image_index) + ".txt", 'w') as f:
            f.write("0 " + str(norm_top_left[0,0]) + " " + str(norm_top_left[0,1])+ " " + str(norm_bot_right[0,0]) + " " + str(norm_bot_right[0,1]))
    image_index+=1

random.seed(0)
# Get the list of all files and directories
path = "C://Users/jrive/Downloads/CCPD2019.tar/CCPD2019/ccpd_base/"

dir_list = os.listdir(path)
random.shuffle(dir_list)

global image_index
image_index = 0
n_files = 3 * len(dir_list)

train_split = int(n_files*0.7)
valid_split = train_split + int(n_files*0.2)
tests_split = valid_split + int(n_files*0.1)

layer = LicencePlateColorGenerator()

def get_coordinates(coords):
    splitted_coords = coords.split("_")
    top_left = np.expand_dims(np.array(splitted_coords[0].split("&")), axis=0).astype(np.float32)
    bot_right = np.expand_dims(np.array(splitted_coords[1].split("&")), axis=0).astype(np.float32)
    return top_left, bot_right

for path_el in dir_list:
    splitted_path = path_el.split("-")
    whole_coord_code = splitted_path[2]
    top_left, bot_right = get_coordinates(whole_coord_code)

    lic_image = Image.open(path + path_el)
    image_size = np.expand_dims(np.array(lic_image.size), axis=0).astype(np.float32)

    norm_top_left = top_left / image_size
    norm_bot_right = bot_right / image_size

    aug = layer(np.asarray(lic_image), training=True)

    for image_n in range(3):
        save_image_and_label(Image.fromarray(aug[image_n]))


    
    


    
    