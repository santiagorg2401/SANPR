import os
import numpy as np
from PIL import Image
import random

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory


class LicencePlateColorGenerator(layers.Layer):
    def __init__(self, **kwargs):
        super(LicencePlateColorGenerator, self).__init__(**kwargs)

    def call(self, images, training=None):
        if not training:
            return images

        yellow_plate = tf.image.adjust_contrast(
            tf.math.multiply(images, -1), 1.4)
        white_plate = tf.image.adjust_saturation(yellow_plate, 0)

        return [images, yellow_plate, white_plate]


def save_image_and_label(lic_image, bbcoords, image_index):
    # Saving the image with a new name
    if image_index <= train_split:
        lic_image.save("/home/santiagorg2401/SANPR-TF/LicensePlates/train/images/ccpd_" + str(image_index) + ".jpg")
        # Saving txt files for labels
        with open("/home/santiagorg2401/SANPR-TF/LicensePlates/train/labels/ccpd_" + str(image_index) + ".txt", 'w') as f:
            f.write("0 " + str(bbcoords[0][0][0]) + " " + str(bbcoords[0][0][1]) + " " + str(
                bbcoords[1]) + " " + str(bbcoords[2]))
    if image_index > train_split and image_index <= valid_split:
        lic_image.save("/home/santiagorg2401/SANPR-TF/LicensePlates/valid/images/ccpd_" + str(image_index) + ".jpg")
        # Saving txt files for labels
        with open("/home/santiagorg2401/SANPR-TF/LicensePlates/valid/labels/ccpd_" + str(image_index) + ".txt", 'w') as f:
            f.write("0 " + str(bbcoords[0][0][0]) + " " + str(bbcoords[0][0][1]) + " " + str(
                bbcoords[1]) + " " + str(bbcoords[2]))
    if image_index > valid_split:
        lic_image.save("/home/santiagorg2401/SANPR-TF/LicensePlates/test/images/ccpd_" + str(image_index) + ".jpg")
        # Saving txt files for labels
        with open("/home/santiagorg2401/SANPR-TF/LicensePlates/test/labels/ccpd_" + str(image_index) + ".txt", 'w') as f:
            f.write("0 " + str(bbcoords[0][0][0]) + " " + str(bbcoords[0][0][1]) + " " + str(
                bbcoords[1]) + " " + str(bbcoords[2]))


def get_coordinates(coords):
    coords = coords.split("_")
    top_left, bot_right = coords[0].split("&"), coords[1].split("&")
    top_left = np.array([int(top_left[0]), int(top_left[1])])
    bot_right = np.array([int(bot_right[0]), int(bot_right[1])])

    return top_left, bot_right


def get_center_widthheight(top_left, bot_right):
    x_center, y_center = (-top_left[0] + bot_right[0]) / 2 + top_left[0], \
                        (-top_left[1] + bot_right[1]) / 2 + top_left[1]
    width, height = bot_right - top_left
    print(width, height)
    return (x_center, y_center), int(width), int(height)


def normalize_bbox(center, width, height, image_shape):
    norm_center = center / image_shape
    norm_width = width/ image_shape[0][0]
    norm_height = height / image_shape[0][1]

    return norm_center, norm_width, norm_height


def run_all(path):
    SEED = 0
    random.seed(SEED)

    # Get the list of all files and directories
    dir_list = os.listdir(path)
    random.shuffle(dir_list)

    image_index = 0
    n_files = 3 * len(dir_list)
    
    global train_split, valid_split, tests_split

    train_split = int(n_files*0.7)
    valid_split = train_split + int(n_files*0.2)
    tests_split = valid_split + int(n_files*0.1)

    layer = LicencePlateColorGenerator()

    for path_el in dir_list:

        splitted_path = path_el.split("-")
        whole_coord_code = splitted_path[2]
        top_left, bot_right = get_coordinates(whole_coord_code)
        centerxy, width, height = get_center_widthheight(top_left, bot_right)
        lic_image = Image.open(path + path_el)
        image_size = np.expand_dims(
            np.array(lic_image.size), axis=0).astype(np.float32)
        norm_center, norm_width, norm_height = normalize_bbox(
            centerxy, width, height, image_size)

        aug = layer(np.asarray(lic_image), training=True)

        for image_n in range(3):
            save_image_and_label(Image.fromarray(
                aug[image_n]), (norm_center, norm_width, norm_height), image_index)
            image_index += 1


def run_once(path):
    SEED = 0
    random.seed(SEED)

    layer = LicencePlateColorGenerator()
    global train_split, valid_split, tests_split
    
    train_split = 5
    valid_split = 10
    tests_split = 15

    image_index = 0
    splitted_path = path.split("-")
    whole_coord_code = splitted_path[2]
    top_left, bot_right = get_coordinates(whole_coord_code)
    centerxy, width, height = get_center_widthheight(top_left, bot_right)
    lic_image = Image.open(path)
    image_size = np.expand_dims(
        np.array(lic_image.size), axis=0).astype(np.float32)
    norm_center, norm_width, norm_height = normalize_bbox(
        centerxy, width, height, image_size)

    aug = layer(np.asarray(lic_image), training=True)

    for image_n in range(3):
        save_image_and_label(Image.fromarray(np.array(
            aug[image_n])), (norm_center, norm_width, norm_height), image_index)
        image_index += 1


if __name__ == "__main__":
    path = "/home/santiagorg2401/CCPD2019/ccpd_base/0130076628352-90_90-273&504_482&579-492&582_268&582_270&504_494&504-0_0_1_11_24_32_26-165-29.jpg"
    run_once(path)

    # path = "/home/santiagorg2401/CCPD2019/ccpd_base/ccpd_"
    # run_once(path)
