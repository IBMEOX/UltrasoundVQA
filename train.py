from options.train_options import TrainOptions
from distutils import dir_util
import shutil
from dataFile import create_dataset
import copy
from models import create_model
import time
from utils.visualizer import Visualizer
import random
import numpy as np
import torch

opt = TrainOptions().parse()

dir_util.copy_tree('./models', opt.checkpoints_dir + '/' + opt.name + '/code/models')
dir_util.copy_tree('./dataFile', opt.checkpoints_dir + '/' + opt.name + '/code/dataFile')
dir_util.copy_tree('./options', opt.checkpoints_dir + '/' + opt.name + '/code/options')
dir_util.copy_tree('./utils', opt.checkpoints_dir + '/' + opt.name + '/code/utils')
shutil.copy(__file__, opt.checkpoints_dir + '/' + opt.name + '/code/')


seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


dataset = create_dataset(opt)
dataset_size = len(dataset)

opt_val = copy.deepcopy(opt)
opt_val.serial_batches = True
dataset_val = create_dataset(opt_val)


print('The number of training images = %d' % dataset_size)

model = create_model(opt)
model.setup(opt)

total_iters = 0
visualizer = Visualizer(opt)

for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0

    print('epoch start')
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if i == 0:
            t_data = iter_start_time - iter_data_time
            print('Data Time Taken: %.2f sec' % (t_data))

        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += 1  # opt.batch_size
        epoch_iter += 1  # opt.batch_size
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
            # save_result = total_iters % opt.update_html_freq == 0
            # model.compute_visuals()
            # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            model.eval()
            model.set_input_val(dataset_val.dataset.__getitem__(i))  # unpack data from data loader
            model.test()  # run inference
            model.train()

            model.compute_visual_result(total_iters)
            print('Save images iters: %d' % (total_iters))


        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)


        if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)


        iter_data_time = time.time()

    if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)
        # model.save_centers('latest')
        # model.save_centers(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()  # update learning rates at the end of every epoch.














