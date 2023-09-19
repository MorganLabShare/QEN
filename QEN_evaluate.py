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


def calculate_qs(tile_path, w_path, json_name, model_file, num_patches, use_gdal, tile_pattern, tile_format):
    tiles_list = glob.glob(os.path.join(tile_path, tile_pattern+tile_format))
    quality_scores = []
    status = -1
    model = keras.models.load_model(model_file)
    for tile in tiles_list:
        if not use_gdal:
            img = k.utils.load_img(tile)
            img = np.asarray(img)
        else:
            ds = gdal.Open(tile, gdal.GA_ReadOnly)
            rb = ds.GetRasterBand(1)
            img = rb.ReadAsArray(0, 0, rb.XSize, rb.YSize)
            img = np.stack((img,) * 3, axis=-1)
        img = 255 - img
        patches, _ = extract_patches(img, num_patches=num_patches, w_path=None, tile_name=None, patch_size=[512, 512], do_save=False)
        data = np.asarray(patches)
        predictions = model.predict(data)
        prediction = str(np.mean(predictions))
        quality_scores.append({'image_name': tile, 'quality_score': prediction})
        print('The quality score for ', tile, ' is ', prediction, '.')

    out_path = os.path.join(w_path, json_name+'.json')
    out_file = open(out_path, 'w')
    json.dump(quality_scores, out_file)
    out_file.close()
    status = 2

    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the quality scores for all tiles listed in the'
                                                 'input json file and create a json file including the quality scores')
    parser.add_argument("--tile_path", default='D:\\KxR_1A\\w5\\w5_2_Sec001_Montage', help='path/to/tiles/')
    parser.add_argument("--write_path", default="./", help='where/to/write/the/output/json/file')
    parser.add_argument('--output_name', default='quality_scores')
    parser.add_argument('--model_file', default="./model_lrDense0001_lrAll001_22_0.000_epoch38.hdf5", help='path/to/model/file')
    parser.add_argument('--num_patches', default=9, help='if it is None, all patches are extracted from the tile image')
    parser.add_argument('--use_gdal', default=True, help='would you like to use gdal package?')
    parser.add_argument('--tile_pattern', type=str, default="*Tile*", help='the pattern in the name of tiles')
    parser.add_argument('--tile_format', default='.tif', type=str, help='define the format of the tile image')

    args = parser.parse_args()

    t1 = time.time()
    status = calculate_qs(tile_path=args.tile_path, w_path=args.write_path, json_name=args.output_name, model_file=args.model_file, num_patches=args.num_patches, use_gdal=args.use_gdal, tile_pattern=args.tile_pattern, tile_format=args.tile_format)
    t2 = time.time()
    if status == 2:
        print('The quality scores are saved in ', os.path.join(args.write_path, args.output_name+'.json'), ', and the total time is ', str(t2-t1), '.')
    else:
        print('Oops! Something went wrong ...')
            
