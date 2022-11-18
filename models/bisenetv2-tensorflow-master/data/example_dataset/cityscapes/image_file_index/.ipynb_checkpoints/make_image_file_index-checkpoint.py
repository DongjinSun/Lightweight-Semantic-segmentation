import os
import os.path as ops
import glob
import random

import tqdm

SOURCE_IMAGE_DIR = './data/icg_drone/dataset/original'
SOURCE_LABEL_DIR = './data/icg_drone/dataset/label'

DST_IMAGE_INDEX_FILE_OUTPUT_DIR = './data/example_dataset/cityscapes/image_file_index'
image_file_index = []

for source_image in tqdm.tqdm(source_image_paths[:240]):
    image_name = ops.split(source_image)[1]
    image_id = image_name.split('.')[0]
    label_image_name = '{:s}.png'.format(image_id)
    label_image_path = ops.join(SOURCE_LABEL_DIR, label_image_name)
    assert ops.exists(label_image_path), '{:s} not exist'.format(label_image_path)
    image_file_index.append('{:s} {:s}'.format(source_image, label_image_path))
random.shuffle(image_file_index)
output_file_path = ops.join(DST_IMAGE_INDEX_FILE_OUTPUT_DIR, 'train.txt')
with open(output_file_path, 'w') as file:
    file.write('\n'.join(image_file_index))

print('Complete')


image_file_index = []

for source_image in tqdm.tqdm(source_image_paths[:320]):
    image_name = ops.split(source_image)[1]
    image_id = image_name.split('.')[0]
    label_image_name = '{:s}.png'.format(image_id)
    label_image_path = ops.join(SOURCE_LABEL_DIR, label_image_name)
    assert ops.exists(label_image_path), '{:s} not exist'.format(label_image_path)

    image_file_index.append('{:s} {:s}'.format(source_image, label_image_path))

random.shuffle(image_file_index)
output_file_path = ops.join(DST_IMAGE_INDEX_FILE_OUTPUT_DIR, 'val.txt')
with open(output_file_path, 'w') as file:
    file.write('\n'.join(image_file_index))

print('Complete')

image_file_index = []

for source_image in tqdm.tqdm(source_image_paths[320:]):
    image_name = ops.split(source_image)[1]
    image_id = image_name.split('.')[0]
    label_image_name = '{:s}.png'.format(image_id)
    label_image_path = ops.join(SOURCE_LABEL_DIR, label_image_name)
    assert ops.exists(label_image_path), '{:s} not exist'.format(label_image_path)

    image_file_index.append('{:s} {:s}'.format(source_image, label_image_path))

random.shuffle(image_file_index)
output_file_path = ops.join(DST_IMAGE_INDEX_FILE_OUTPUT_DIR, 'test.txt')
with open(output_file_path, 'w') as file:
    file.write('\n'.join(image_file_index))

print('Complete')