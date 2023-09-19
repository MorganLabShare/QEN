import argparse
import json
import os.path
import keras
import glob
import numpy as np
import PIL.Image
from tensorflow import keras as k
import time
from utilities import extract_patches
from osgeo import gdal

PIL.Image.MAX_IMAGE_PIXELS = 500000000


def calculate_qs(tile_path, model, num_patches=9, use_gdal=True):
    tilet1 = time.time()
    if not use_gdal:
        img = k.utils.load_img(tile_path)
        img = np.asarray(img)
    else:
        ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
        rb = ds.GetRasterBand(1)
        img = rb.ReadAsArray(0, 0, rb.XSize, rb.YSize)
        img = np.stack((img,) * 3, axis=-1)
    img = 255 - img
    tilet2 = time.time()
    patches, indexes = extract_patches(img, num_patches=num_patches, w_path=None, tile_name=None, patch_size=[512, 512],
                                       do_save=False)
    data = np.asarray(patches)
    tile_name = os.path.basename(tile_path)
    tilet3 = time.time()
    predictions = model.predict(data)
    tilet4 = time.time()
    print('total time for ', tile_name, ' is ', str(tilet2 - tilet1), ' secs for reading and ', str(tilet3 - tilet2),
          ' secs for sampling and ', str(tilet4 - tilet3), ' secs for QS calculation.')

    return np.mean(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the quality scores for all tiles listed in the'
                                                 'input json file and create a json file including the quality scores')
    parser.add_argument("--tiles_path", default='D:\\KxR_1A\\w5\\w5_2_Sec001_Montage', help='path/to/tiles/')
    parser.add_argument("--write_path",
                        default="\\\\storage1.ris.wustl.edu\\jlmorgan\\Active\\morganLab\\PUBLICATIONS\\EMqual\\Codes\\",
                        help='where/to/write/the/output/json/file')
    parser.add_argument('--output_name', default='quality_scores_test_gdal')
    parser.add_argument('--model_file', default="./model_lrDense0001_lrAll001_22_0.000_epoch38.hdf5",
                        help='path/to/model/file')
    parser.add_argument('--num_patches', default=9, help='if it is None, all patches are extracted from the tile image')
    parser.add_argument('--use_gdal', default=True, help='would you like to use gdal package?')
    parser.add_argument('--tile_pattern', type=str, default="*Tile*", help='the pattern in the name of tiles')
    parser.add_argument('--tile_format', default='.tif', type=str, help='define the format of the tile image')

    args = parser.parse_args()

    quality_scores = []
    t1 = time.time()
    tiles_list = glob.glob(os.path.join(args.tiles_path, args.tile_pattern + args.tile_format))
    model = keras.models.load_model(args.model_file)
    for tile in tiles_list:
        quality_score = calculate_qs(tile_path=tile, model=model,
                                     num_patches=args.num_patches, use_gdal=args.use_gdal)
        quality_scores.append({'image_name': tile, 'quality_score': quality_score})

    out_path = os.path.join(args.write_path, args.output_name + '.json')
    out_file = open(out_path, 'w')
    json.dump(quality_scores, out_file)
    out_file.close()
    t2 = time.time()
    print('total time is: ', str(t2 - t1))
