import json
import os
import time
from predict_quality_scores import calculate_qs
import keras
import openpyxl
import argparse
import numpy as np
import datetime
from tabulate import tabulate


def read_items_from_file(txt_path, mode='r'):
    items = []
    with open(txt_path, mode) as file:
        items = json.load(file)
    file.close()
    return items


def get_items(items_path):
    items = []
    for root, dirs, files in os.walk(items_path):
        for item in dirs + files:
            items.append(os.path.join(root, item))
    return items


def find_new_tiles(directory, pattern, model, workbook, previous_items, tile_format, use_gdal, num_patches):
    current_items = get_items(directory)
    sheet = workbook.active
    items = []
    for item in previous_items:
        items.append(item['item_name'])
    new_items = set(current_items) - set(items)
    txt_path = os.path.join(directory, "qualResult.json")
    items = previous_items

    for item in new_items:
        if item.lower().endswith(tile_format):
            quality_score = calculate_qs(tile_path=item, model=model, use_gdal=use_gdal,
                                         num_patches=num_patches)
            sheet.append((item, quality_score))
            items.append({'item_name': item, 'quality_score': str(quality_score), 'datetime': str(datetime.datetime.now())})
        else:
            items.append({'item_name': item, 'quality_score': 'None', 'datetime': str(datetime.datetime.now())})
    workbook.save(os.path.join(directory, "quality_scores.xlsx"))
    with open(txt_path, 'w') as file:
        json.dump(items, file)
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script calculates the quality scores for all tiles in the'
                                                 'input folder and create a spreadsheet including the quality scores')
    parser.add_argument("--tiles_path", default='D:\\test', help='path/to/tiles/')
    parser.add_argument('--model_file', default="./model_lrDense0001_lrAll001_22_0.000_epoch38.hdf5",
                        help='path/to/model/file')
    parser.add_argument('--num_patches', default=9, help='if it is None, all patches are extracted from the tile image')
    parser.add_argument('--use_gdal', default=True, help='would you like to use gdal package?')
    parser.add_argument('--tile_pattern', type=str, default="Tile", help='the pattern in the name of tiles')
    parser.add_argument('--tile_format', default='.png', type=str, help='define the format of the tile image')

    args = parser.parse_args()

    sheet_path = os.path.join(args.tiles_path, "quality_scores.xlsx")
    if os.path.exists(sheet_path):
        workbook = openpyxl.load_workbook(sheet_path)
    else:
        workbook = openpyxl.Workbook()

    model = keras.models.load_model(args.model_file)

    txt_path = os.path.join(args.tiles_path, "qualResult.json")
    previous_items = []
    if os.path.exists(txt_path):
        previous_items = read_items_from_file(txt_path)
    model = keras.models.load_model(args.model_file)

    try:
        while True:
            find_new_tiles(directory=args.tiles_path, pattern=args.tile_pattern, model=model,
                           workbook=workbook,
                           previous_items=previous_items, tile_format=args.tile_format,
                           num_patches=args.num_patches,
                           use_gdal=args.use_gdal)
            previous_items = read_items_from_file(txt_path)
            time.sleep(5)
    except KeyboardInterrupt:
        workbook.close()
