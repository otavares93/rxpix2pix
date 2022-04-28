"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import stats
import pandas as pd
import numpy as np


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot_aux = opt.dataroot
    opt.name_aux = opt.name
    for fold in range(opt.n_folds):
        opt.test = fold
        if opt.isTB:
            path_div_ext = '_tb_cluster/KL_JS.csv'
            path_norm_ext = '_tb_cluster/L1_L2.csv'
            opt.name = opt.name_aux + str(fold) + '_tb_cluster'
        else:
            path_div_ext = '_ntb_cluster/KL_JS.csv'
            path_norm_ext = '_ntb_cluster/L1_L2.csv'
            opt.name = opt.name_aux + str(fold) + '_ntb_cluster'
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        div = {
            'model_iter': [],
            'js_rr': [],
            'js_rf': [],
            'kl_rr': [],
            'kl_rf': [],
        }
        norm_l1_l2 = {
            'model_iter': [],
            'l1_rr': [],
            'l1_rf': [],
            'l2_rr': [],
            'l2_rf': [],
        }
        for model_iter in [200, 400, 600, 800, 1000]:
            opt.load_iter = model_iter
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            # initialize logger
            if opt.use_wandb:
                wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
                wandb_run._label(repo='CycleGAN-and-pix2pix')

            # create a website
            web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
            if opt.load_iter > 0:  # load_iter is 0 by default
                web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
            print('creating web directory', web_dir)
            webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
            # test with eval mode. This only affects layers like batchnorm and dropout.
            # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
            # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
            if opt.eval:
                model.eval()

            real_imgs = []
            fake_imgs = []
            for i, data in enumerate(dataset):
                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                visuals = model.get_current_visuals()  # get image results
                realB = visuals['real_B']
                fakeB = visuals['fake_B']
                real_imgs.append(torch.flatten(realB).detach().cpu().numpy())
                fake_imgs.append(torch.flatten(fakeB).detach().cpu().numpy())
                img_path = model.get_image_paths()     # get image paths
                if i % 5 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
            val_kl_rr, val_js_rr = stats.calculate_divergences( np.array( real_imgs ) , np.array( real_imgs ))
            val_kl_rf, val_js_rf = stats.calculate_divergences( np.array( real_imgs ) , np.array( fake_imgs ))
            val_l1_rr, val_l2_rr = stats.calculate_l1_and_l2_norm_errors(  np.array( real_imgs )  ,   np.array( real_imgs ) )
            val_l1_rf, val_l2_rf = stats.calculate_l1_and_l2_norm_errors(  np.array( real_imgs )  ,   np.array( fake_imgs ) )
            #calculating js and kl divergences
            div['model_iter'].append(model_iter)
            div['js_rr'].append(val_js_rr)
            div['js_rf'].append(val_js_rf)
            div['kl_rr'].append(val_kl_rr)
            div['kl_rf'].append(val_kl_rf)
            #calculating l1 and l2 norms
            norm_l1_l2['model_iter'].append(model_iter)
            norm_l1_l2['l1_rr'].append(val_l1_rr)
            norm_l1_l2['l1_rf'].append(val_l1_rf)
            norm_l1_l2['l2_rr'].append(val_l2_rr)
            norm_l1_l2['l2_rf'].append(val_l2_rf)

            webpage.save()  # save the HTML
        path_div_csv = './results/fold' + str(fold) + path_div_ext
        path_norm_csv = './results/fold' + str(fold) + path_norm_ext
        pd.DataFrame(div).to_csv(path_div_csv)
        pd.DataFrame(norm_l1_l2).to_csv(path_norm_csv)
