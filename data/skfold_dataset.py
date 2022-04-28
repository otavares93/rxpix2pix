"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from util.stratified_kfold import stratified_train_val_test_splits_bins
from util.util import prepare_my_table
import pandas as pd
#import numpy as np
import os, sys
#import glob
#import re
#import hashlib
#import pathlib
#import cv2
# from data.image_folder import make_dataset
# from PIL import Image


class SkfoldDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    #@staticmethod
    #def modify_commandline_options(parser, is_train):
    #    """Add new dataset-specific options, and rewrite default values for existing options.

    #    Parameters:
    #        parser          -- original option parser
    #        is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    #    Returns:
    #        the modified parser.
    #    """
    #    #parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
    #    #parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        #return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        ####

        #clinical_path = opt.dataroot + '/ClinicalReadings'
        if opt.isTrain is False:
            opt.train_dataset = False
        clinical_path = opt.dataroot + '/raw/clinical'
        if opt.custom_masks_path is None:
            masks_path = opt.dataroot + '/A'
        else:
            masks_path = opt.custom_masks_path
        if opt.custom_images_path is None:
            images_path = opt.dataroot + '/B'
        else:
            images_path = opt.custom_images_path
        if opt.custom_paired_path is None:
            paired_path = opt.dataroot + '/AB'
        else:
            paired_path = opt.custom_paired_path

        if opt.generate_paths_data_csv:
            df = prepare_my_table(clinical_path, images_path, masks_path, paired_path, combine=opt.dataset_action)
            df.to_csv('Shenzhen_pix2pix_table_from_raw.csv')
        else:
            df = pd.read_csv('Shenzhen_pix2pix_table_from_raw.csv')
        splits = stratified_train_val_test_splits_bins(df, opt.n_folds, opt.seed)[opt.test]
        training_data = df.iloc[splits[opt.sort][0]]
        validation_data = df.iloc[splits[opt.sort][1]]

        if (opt.isTB == True):
            self.train_tb = training_data.loc[df.target == 1]
            self.val_tb = validation_data.loc[df.target == 1]
            #print(type(self.train_tb['paired_image_path']))
            if opt.train_dataset:
                self.AB_paths = [img_path for img_path in self.train_tb['paired_image_path'].tolist() if img_path != '']
            else:
                self.AB_paths = [img_path for img_path in self.val_tb['paired_image_path'].tolist() if img_path != '']
        else:
            self.train_ntb = training_data.loc[df.target == 0]
            self.val_ntb = validation_data.loc[df.target == 0]
            if opt.train_dataset:
                self.AB_paths = [img_path for img_path in self.train_ntb['paired_image_path'].tolist() if img_path != '']
            else:
                self.AB_paths = [img_path for img_path in self.val_ntb['paired_image_path'].tolist() if img_path != '']
        # get the image paths of your dataset;
         # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        #print(self.AB_paths)
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)

        ####
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        #base_data_raw_path = '/Users/ottotavares/Documents/COPPE/projetoTB/China/CXR_png/unaligned'

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
