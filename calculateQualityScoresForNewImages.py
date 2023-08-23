import os
import time
from predict_quality_scores import calculate_qs
import keras
import openpyxl


def find_new_tiles(directory, pattern, model, workbook, existing_files):
    sheet = workbook.active
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif') and file.startswith(pattern) and file not in existing_files:
                existing_files.add(file)
                file_path = os.path.join(root, file)
                quality_score = calculate_qs(tile_path=file_path, model=model)
                sheet.append((file, quality_score))
                workbook.save(os.path.join(directory, "quality_scores.xlsx"))


if __name__ == "__main__":
    directory_to_watch = 'D:\\KxR_1A\\w5\\w5_2_Sec001_Montage'  # set in matlab script
    file_pattern = "Tile"  # set in matlab script
    model_file = "./model_lrDense0001_lrAll001_22_0.000_epoch38.hdf5"  # set in matlab script
    sheet_path = os.path.join(directory_to_watch, "quality_scores.xlsx")
    if os.path.exists(sheet_path):
        workbook = openpyxl.load_workbook(sheet_path)
    else:
        workbook = openpyxl.Workbook()

    model = keras.models.load_model(model_file)
    existing_files = set()

    try:
        while True:
            find_new_tiles(directory_to_watch, file_pattern, model, workbook, existing_files)
            time.sleep(30)
    except KeyboardInterrupt:
        workbook.close()

