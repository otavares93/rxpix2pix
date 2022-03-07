"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import numpy as np
import torch
from options.train_options import TrainOptions
#from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.stats import calculate_divergences, calculate_l1_and_l2_norm_errors

#def calculate_divergences( real_samples, fake_samples ):
#  kl, js = calculate_divergences(real_samples, fake_samples)
#  return np.mean(kl), np.mean(js)

#def calculate_l1_and_l2_norm_errors( real_samples, fake_samples):
#  l1, l2 = calculate_l1_and_l2_norm_errors( real_samples, fake_samples )
#  return np.mean(l1),  np.mean(l2)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #opt.train_dataset = False      #fliping train dataset flag for defining val dataset
    #dataset_val = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #opt.train_dataset = True       #fliping back train dataset flag for train
    dataset_size = len(dataset)    # get the number of images in the dataset.
    #dataset_val_size = len(dataset_val)  # get the number of images in the dataset.
    #opt_val = TestOptions().parse()
    # dataset_val = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % dataset_size)
    #print('The number of val images = %d' % dataset_val_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)


            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                print(losses)
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        #if total_iters % opt.val_freq == 0: # calling validation routine for evaluating the generator
        #  real_imgs = []
        #  fake_imgs = []
        #  for i, data_val in enumerate(dataset_val):
        #    if i >= opt.num_val:
        #      break
        #    model.set_input(data_val)  # unpack data from data loader
        #    model.test()
        #    visuals_val = model.get_current_visuals()  # get image results
        #    realB = visuals_val['real_B']
        #    fakeB = visuals_val['fake_B']
        #    real_imgs.append(torch.flatten(realB).detach().cpu().numpy())
        #    fake_imgs.append(torch.flatten(fakeB).detach().cpu().numpy())

        #  val_kl_rr, val_js_rr = calculate_divergences( np.array( real_imgs ) , np.array( real_imgs ))
        #  val_kl_rf, val_js_rf = calculate_divergences( np.array( real_imgs ) , np.array( fake_imgs ))
        #  print(np.mean(val_kl_rr))
        #  print(np.mean(val_kl_rf))

        iter_data_time = time.time()
        print('end of validation step')
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
