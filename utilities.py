
import math
import os
import numpy as np
from tensorflow import keras as k
import random


def extract_patches(img, num_patches=None, w_path=None, tile_name=None, patch_size=[512, 512], do_save=False):

    stride = patch_size
    img_size = np.shape(img)
    height, width = img_size[0], img_size[1]
    all_indexes = []
    
    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            all_indexes.append({'y': y, 'x': x})

    indexes = all_indexes
    if num_patches is not None:
        indexes = random.sample(all_indexes, num_patches)

    patches = []
    for index in indexes:
        y, x = index['y'], index['x']
        patch = img[y:y + patch_size[0], x:x + patch_size[1], :]
        patches.append(patch)
##            if do_save:
##                k.utils.save_img(os.path.join(w_path, os.path.splitext(tile_name)[0]+'_y'+str(y)+'_x'+str(x)+'.png'), patch)

    return patches, indexes




