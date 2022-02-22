
import os
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
#from data.image_folder import make_dataset
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import json

import pandas as pd
import numpy as np
import os, sys
import glob
import re
import hashlib
import pathlib
import cv2


#TO-DO acrescentar isTB no options
isTB = True

#from options.train_options import TrainOptions
#from data import create_dataset
#from models import create_model
#from rxwgan.models import *
#from rxwgan.wgangp import wgangp_optimizer
#from rxcore import stratified_train_val_test_splits

def run(command: object) -> object:
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def expand_folder( path , extension):
    l = glob.glob(path+'/*.'+extension)
    l.sort()
    return l

def get_md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()

#import numpy as np


#
# Split train/val/test splits
#
def stratified_train_val_test_splits( df_kfold, seed=512 ):

    cv_train_test = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_train_val  = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    sorts_train_test = []

    for train_val_idx, test_idx in cv_train_test.split( df_kfold.values,df_kfold.target.values ):
        train_val_df = df_kfold.iloc[train_val_idx]
        sorts = []
        for train_idx, val_idx in cv_train_val.split( train_val_df.values, train_val_df.target.values ):
            sorts.append((train_val_df.index[train_idx].values, train_val_df.index[val_idx].values, test_idx))
        sorts_train_test.append(sorts)
    return sorts_train_test

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def prepare_my_table(clinical_path, images_path, masks_path, combine = False):
    d = {
        'target': [],
        'image_ID': [],
        'raw_image_path': [],
        'mask_image_path': [],
        'paired_image_path': [],
        'raw_image_md5': [],
        'age': [],
        'sex': [],
        'comment': [],
    }

    def treat_string(lines):
        string = ''
        for s in lines:
            string += s.replace('\n', '').replace('\t', '')
        return re.sub(' +', ' ', string)

    for idx, path in enumerate(expand_folder(clinical_path, 'txt')):
        with open(path, 'r') as f:
            lines = f.readlines()
            sex = 'male' if 'male' in lines[0] else 'female'  # 1 for male and 0 for female
            age = int(re.sub('\D', '', lines[0]))
            # get TB by file name (_1.txt is PTB or _0.txt is NTB)
            target = 1 if '_1.txt' in path else 0

            filename = path.split('/')[-1]
            image_filename = filename.replace('txt', 'png')
            # image_path = images_path+('/tb/' if target else '/no_tb/')+image_filename
            image_path = images_path + '/' + image_filename
            d['target'].append(target)
            d['age'].append(age)
            d['sex'].append(sex)
            d['raw_image_path'].append(image_path)
            d['raw_image_md5'].append(get_md5(image_path))
            d['mask_image_path'].append('')
            d['paired_image_path'].append('')
            d['comment'].append(treat_string(lines[1::]))
            d['image_ID'].append(filename.replace('.txt', ''))
            l_masks = make_dataset(masks_path)
            for mask in l_masks:
                if image_path[-17:] == mask[-17:]:
                    idx = np.where(np.array(d['raw_image_path']) == image_path)[0][0]
                    d['mask_image_path'][idx] = mask
                    if combine == True:
                        path_paired = image_path[:-25] + 'foldAB'
                        path_paired_img = path_paired + '/' + image_path[-17:]
                        d['paired_image_path'][idx] = path_paired_img
                        if not os.path.isdir(path_paired):
                            os.makedirs(path_paired)
                        im_A = cv2.imread(image_path)
                        im_B = cv2.imread(mask)
                        im_AB = np.concatenate([im_B, im_A], 1)
                        cv2.imwrite(path_paired_img, im_AB)

    return pd.DataFrame(d)


# NOTE: this is optional.
#from rxcore import allow_tf_growth
#allow_tf_growth()

#
# Start your job here
#

#job  = json.load(open(args.job, 'r'))
#sort = job['sort']
#target = 1 # tb active
#test = job['test']
seed = 512
#epochs = 1000
#batch_size = 32

base_data_raw_path = '/Users/ottotavares/Documents/COPPE/projetoTB/China/CXR_png/unaligned'
clinical_path = base_data_raw_path + '/ClinicalReadings'
images_path = base_data_raw_path + '/trainA'
masks_path = base_data_raw_path + '/trainB'

df = prepare_my_table(clinical_path, images_path, masks_path, combine = True)

splits = stratified_train_val_test_splits(df,seed)[0]
training_data   = df.iloc[splits[0][0]]
validation_data = df.iloc[splits[0][1]]

if(isTB == True):
    train_tb = training_data.loc[df.target==1]
    val_tb = validation_data.loc[df.target==1]
else:
    train_ntb = training_data.loc[df.target==0]
    val_ntb = validation_data.loc[df.target == 0]

#training_data = training_data.loc[training_data.target==target]
#validation_data = validation_data.loc[validation_data.target==target]

extra_d = {'sort' : sort, 'test':test, 'target':target, 'seed':seed}


# Run!
#history = optimizer.fit( train_generator , val_generator, extra_d=extra_d, wandb=wandb )
combine_ab = 'python datasets/combine_A_and_B.py --fold_A /Users/ottotavares/Documents/COPPE/projetoTB/China/CXR_png/unaligned/trainA --fold_B /Users/ottotavares/Documents/COPPE/projetoTB/China/CXR_png/unaligned/trainB --fold_AB /Users/ottotavares/Documents/COPPE/projetoTB/China/CXR_png/unaligned'
run(combine_ab)
# pix2pix train/test
#train_cmd = 'python train.py --model pix2pix --name ' + 'test_%d_sort_%d'%(test,sort) + '--dataroot . --n_epochs 1 --n_epochs_decay 5 --save_latest_freq 10 --display_id -1'
#run(train_cmd)


#run('python test.py --model pix2pix --name temp_pix2pix --dataroot ./datasets/mini_pix2pix --num_test 1')


