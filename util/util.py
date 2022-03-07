"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


from data.image_folder import make_dataset
import pandas as pd
import numpy as np
import glob
import re
import hashlib
import pathlib
import cv2


###Pix2pix custom helper functions
def expand_folder( path , extension):
    l = glob.glob(path+'/*.'+extension)
    l.sort()
    return l

def get_md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()

def prepare_my_table(clinical_path, images_path, masks_path, path_paired, combine = False):
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
                    #print(image_path[-17:])
                    #print(mask[-17:])
                    d['mask_image_path'][idx] = mask
                    if path_paired is None:
                        path_paired = image_path[:-19] + 'AB'
                    path_paired_img = path_paired + '/' + image_path[-17:]
                    #print(path_paired_img)
                    d['paired_image_path'][idx] = path_paired_img
                    if combine == True:
                        if not os.path.isdir(path_paired):
                            os.makedirs(path_paired)
                        im_B = cv2.imread(image_path)
                        im_A = cv2.imread(mask)
                        im_AB = np.concatenate([im_A, im_B], 1)
                        cv2.imwrite(path_paired_img, im_AB)

    return pd.DataFrame(d)

###Pix2pix default helper functions
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
