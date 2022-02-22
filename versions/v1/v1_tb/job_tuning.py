#!/usr/bin/env python3
__all__ = ['stratified_train_val_test_splits']
# NOTE: mandatory
try:
    from orchestra import complete, start, fail, is_test_job
    is_job = True
    is_test = is_test_job()
except:
    is_job = False
    is_test = False

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
#from data.image_folder import make_dataset
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import json
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



#import numpy as np

# NOTE: this is optional.
#from rxcore import allow_tf_growth
#allow_tf_growth()


#
# Input args (mandatory!)
#
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-v','--volume', action='store',
    dest='volume', required = False,
    help = "volume path")

parser.add_argument('-i','--input', action='store',
    dest='input', required = True, default = 1,
    help = "Input image directory.")

parser.add_argument('-j','--job', action='store',
    dest='job', required = True, default = 1,
    help = "job configuration.")

parser.add_argument('--disable_wandb', action='store_true',
    dest='disable_wandb', required = False,
    help = "Disable wandb report")



import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()
print(args)


try:

    if is_job: start()

    #
    # Start your job here
    #

    job  = json.load(open(args.job, 'r'))
    sort = job['sort']
    target = 1 # tb active
    test = job['test']
    seed = 512
    epochs = 1000
    batch_size = 32

    output_dir = args.volume + '/test_%d_sort_%d'%(test,sort)

    #
    # Check if we need to recover something...
    #
    if os.path.exists(output_dir+'/checkpoint.json'):
        print('reading from last checkpoint...')
        checkpoint = json.load(open(output_dir+'/checkpoint.json', 'r'))
        #history = json.load(open(checkpoint['history'], 'r'))
        #critic = tf.keras.models.load_model(checkpoint['critic'])
        #generator = tf.keras.models.load_model(checkpoint['generator'])
        start_from_epoch = checkpoint['epoch'] + 1
        continue_train = True
        print('starts from %d epoch...'%start_from_epoch)

    else:
        start_from_epoch= 0
        continue_train = False
        # create models
        #critic = Critic_v1().model
        #generator = Generator_v1().model
        #history = None

    #height = critic.layers[0].input_shape[0][1]
    #width  = critic.layers[0].input_shape[0][2]

    # Read dataframe
    #dataframe = pd.read_csv(args.input)
    #l_imgs = make_dataset(args.input)
    #dataframe = pd.DataFrame(l_imgs, columns=['imgs'])

    #splits = stratified_train_val_test_splits(dataframe,seed)[test]
    #training_data   = dataframe.iloc[splits[sort][0]]
    #validation_data = dataframe.iloc[splits[sort][1]]

    #training_data = training_data.loc[training_data.target==target]
    #validation_data = validation_data.loc[validation_data.target==target]

    #extra_d = {'sort' : sort, 'test':test, 'target':target, 'seed':seed}

    # image generator
    #datagen = ImageDataGenerator( rescale=1./255 )

    #train_generator = datagen.flow_from_dataframe(training_data, directory = None,
                                                  #x_col = 'raw_image_path',
                                                  #y_col = 'target',
                                                  #batch_size = batch_size,
                                                  #target_size = (height,width),
                                                  #class_mode = 'raw',
                                                  #shuffle = True,
                                                  #color_mode = 'grayscale')

    #val_generator   = datagen.flow_from_dataframe(validation_data, directory = None,
     #                                             x_col = 'raw_image_path',
    #                                             y_col = 'target',
     #                                             batch_size = batch_size,
    #                                             class_mode = 'raw',
     #                                             target_size = (height,width),
    #                                             shuffle = True,
     #                                             color_mode = 'grayscale')

    #
    # Create optimizer
    #

    #optimizer = wgangp_optimizer( critic, generator,
    #                              n_discr = 0,
     #                             history = history,
    #                             start_from_epoch = 0 if is_test else start_from_epoch,
     #                             max_epochs = 1 if is_test else epochs,
    #                             output_dir = output_dir,
     #                             disp_for_each = 50,
    #                             save_for_each = 50 )


    #try:
    #    if args.disable_wandb:
    #        wandb=None
    #    else:
    #        import wandb
    #        task = args.volume.split('/')[-2]
    #        name = 'test_%d_sort_%d'%(test,sort)
    #        wandb.init(project=task,
    #                   name=name,
    #                   id=name)

    #except:
    #    wandb=None


    # Run!
    #history = optimizer.fit( train_generator , val_generator, extra_d=extra_d, wandb=wandb )

    # pix2pix train/test
    train_cmd = 'python train.py --dataroot /content/gdrive/MyDrive/pix2pix/data/unaligned --model pix2pix --dataset_mode skfold --direction AtoB --name ' + 'rxpix2pix_test_%d_sort_%d'%(test,sort) + ' --use_wandb --wandb_fold_id ' + 'rxpix2pix_test_%d_sort_%d'%(test,sort) + ' --gpu_ids 0 --val_freq 100 --num_val 100 --n_epochs 500 --n_epochs_decay 500 --test ' + str(test) + ' --sort ' + str(sort)
    run(train_cmd)
    #run('python test.py --model pix2pix --name temp_pix2pix --dataroot ./datasets/mini_pix2pix --num_test 1')

    # in the end, save all by hand
    #critic.save(output_dir + '/critic_trained.h5')
    #generator.save(output_dir + '/generator_trained.h5')
    #with open(output_dir+'/history.json', 'w') as handle:
    #  json.dump(history, handle,indent=4)

    #
    # End your job here
    #

    if is_job: complete()
    sys.exit(0)

except  Exception as e:
    print(e)
    # necessary to work on orchestra
    if is_job: fail()
    sys.exit(1)
