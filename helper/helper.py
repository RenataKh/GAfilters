import numpy as np
import os
from PIL import Image
import scipy.misc

# File processing function:

def get_images_names_from_folder(folder_name):
    return [folder_name + i for i in os.listdir(folder_name) if (i[-3:] == 'jpg' or i[-3:] == 'png')]
 
def get_image_as_array(filename, resolution=1, process_image=True):
    im = Image.open(filename)
    im = np.array(im, dtype='float32')
    im_array = im.reshape(im.shape[0], im.shape[1], -1)
    if process_image and resolution != 1:
        return scipy.misc.imresize(im_array, (im.shape[0] / resolution, im.shape[1] / resolution, 3), interp='bilinear')
    im_array = np.array(im_array, dtype='float32')
    return im_array


def get_matrix_of_images(image_file_names):
    return np.array([get_image_as_array(name) for name in image_file_names])