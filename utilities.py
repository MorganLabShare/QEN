
import math
import os
import numpy as np
from tensorflow import keras as k
import tensorflow as tf
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


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=20, num_class=4, crop_size=[224, 224], shuffle=True):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.crop_size = crop_size
        # self.preprocess = preprocess
        self.shuffle = shuffle
        self.datalen = len(self.x_set)
        self.indexes = np.arange(self.datalen)
        self.num_class = num_class
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.datalen/self.batch_size)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.x_set[i] for i in batch_indexes]
        image_x = np.empty((len(batch_x), *self.crop_size, 3))
        # label_y = []
        # label_y = np.zeros((len(batch_x), self.num_class))
        # if self.y_set is not None:
        batch_y = [self.y_set[i] for i in batch_indexes]
        label_y = np.zeros((len(batch_y), self.num_class))
        i = 0

        for file_name in batch_x:
            img = tf.keras.utils.load_img(file_name)
            img = crop_image(img, self.crop_size)
            if img is not None:
                image_x[i, ] = img
                if self.num_class > 1:
                    label_y[i, batch_y[i]] = 1
                else:
                    label_y[i] = batch_y[i]
                i += 1
        # image_x = self.preprocess(image_x)
        return image_x, label_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)



