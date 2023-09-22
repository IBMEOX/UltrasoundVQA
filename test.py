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
from options.test_options import TestOptions
from dataFile import create_dataset
from models import create_model
from utils.visualizer import save_image
import scipy.io as sio
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, accuracy_score, roc_auc_score,\
    confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def calc_score(label, all_score):

    [Pre, Rec, thresholds] = precision_recall_curve(label, all_score)
    F1 = 2 * Pre * Rec / (Pre + Rec +1e-20)
    thres = thresholds[F1.argmax()]
    labelPred = all_score.copy()
    labelPred[all_score > thres], labelPred[all_score <= thres] = 1, 0
    result = precision_recall_fscore_support(label, labelPred, average='binary')
    acc = accuracy_score(label, labelPred)
    auc = roc_auc_score(label, all_score)
    f1, pre, rec = result[2], result[0], result[1]
    tn, fp, fn, tp = confusion_matrix(label, labelPred).ravel()

    return f1, pre, rec, auc, acc, tn, fp, fn, tp, labelPred, thres


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    
    opt.input_nc = 1
    opt.output_nc = 1
    # opt.load_iter = 0
    # opt.downsample = 4
    # opt.ngf = 32
    # opt.ndf = 64
    opt.serial_batches = True
    # opt.netG_A = 'encoder'
    # opt.netG_B = 'decoder'

    # opt.eval = True
    if opt.name == 'experiment_name':
        opt.name = 'test'
    result_name = 'result_z'
    save_name = result_name + '.npz'
    # if 'map' not in opt.name:
    #     opt.latent = 'vector'
    #     opt.nz = 1024


    saveFlag = True

    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of test images = %d' % len(dataset))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model.load_centers(opt)
    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    web_dir = os.path.join(opt.results_dir, opt.name, 'results')  # define the website directory

    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)
    if not os.path.exists(web_dir +'/images'):
        os.makedirs(web_dir+'/images')
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    # with open(web_dir + '/' + result_name + '.txt', 'w') as f:
    #     f.close()

    dataMM= dataset.dataset

    IMG_score, IMG_score_0, IMG_score_1, IMG_score_2, IMG_score_3 = [], [], [], [], []
    Z_score, Z_score_0, Z_score_1, Z_score_2, Z_score_3 = [], [], [], [], []
    label = []
    SSIM_score, SSIM_score_0, SSIM_score_1, SSIM_score_2, SSIM_score_3 = [], [], [], [], []
    CENTER_score = []

    model.eval()

    cls_1_num, cls_2_num, cls_0_num, cls_3_num = 0, 0, 0, 0
    tst_num = 3000


    #import pdb; pdb.set_trace()
    if not os.path.isfile(web_dir +'/' + save_name):
        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            #     break
            # i=i+9100
            handover = False
            if handover:
                # dataPick = dataMM.__getitem__(i)
                # dataTarget = dataMM.targets[i]
                # dataPick = dataMM.transform_A(dataMM.np_to_img(dataPick)).unsqueeze(0)
                # data['A'] = dataPick
                data['A'] = dataMM.__getitem__(i)['A'].unsqueeze(0)
                data['B'] = dataMM.__getitem__(i)['B'].unsqueeze(0)
                data['A_label'] = torch.tensor([dataMM.__getitem__(i)['A_label']])


            dataTarget = data['A_label'].cpu().numpy()[0]
            dataPath = data['A_paths'][0].split('/')[-1]

            if dataTarget == 0 and cls_0_num < tst_num:
                cls_0_num += 1
            elif dataTarget == 1 and cls_1_num < tst_num:
                cls_1_num += 1
            elif dataTarget == 2 and cls_2_num < tst_num:
                #continue
                cls_2_num += 1
            elif dataTarget == 3 and cls_3_num < tst_num:
                cls_3_num += 1
            elif cls_0_num >= tst_num and cls_1_num >= tst_num and cls_2_num >= tst_num and cls_3_num >= tst_num:
                break
            else:
                continue
            
            # if cls_0_num >= tst_num or cls_1_num >= tst_num or cls_2_num >= tst_num or cls_3_num >= tst_num :
            #     continue
            # elif cls_0_num >= tst_num and cls_1_num >= tst_num and cls_2_num >= tst_num and cls_3_num >= tst_num:
            #     break
            


            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference

            visuals = model.get_current_visuals()  # get image results

            img_score = torch.nn.L1Loss()(visuals['real_A'], visuals['rec_A']).cpu().numpy()
            z_score = torch.nn.L1Loss()(visuals['fake_B'],visuals['ad_B']).cpu().numpy()

            try:
                #import pdb
                #pdb.set_trace()    
                center_score = torch.nn.L1Loss()(visuals['fake_B'].cpu().view(-1), model.centers[0].view(-1)).numpy()
            except:
                center_score = 0

            tem_img = np.array(IMG_score.copy())
            tem_label = np.array(label.copy())
            tem_score = tem_img[np.argwhere(tem_label==dataTarget)]

            img, ssim_score = model.compute_visual_result(visuals=visuals)

            IMG_score.append(img_score)
            Z_score.append(z_score)
            label.append(dataTarget)
            SSIM_score.append(ssim_score)
            CENTER_score.append(center_score)

            if dataTarget == 0:
                IMG_score_0.append(img_score)
                Z_score_0.append(z_score)
                SSIM_score_0.append(ssim_score)
            elif dataTarget == 1:
                IMG_score_1.append(img_score)
                Z_score_1.append(z_score)
                SSIM_score_1.append(ssim_score)
            elif dataTarget == 2:
                IMG_score_2.append(img_score)
                Z_score_2.append(z_score)
                SSIM_score_2.append(ssim_score)
            elif dataTarget == 3:
                IMG_score_3.append(img_score)
                Z_score_3.append(z_score)
                SSIM_score_3.append(ssim_score)

            if saveFlag:
                save_image(img, [2, 4], web_dir + '/images/' + str(dataTarget) + '_' + str(i) + '.png')

            # fea_score = (model.fea_gen(net=model.netD_A, input=visuals['fake_B'], slice=6)[0][0].data -
            #  model.fea_gen(net=model.netD_A, input=visuals['ad_B'], slice=6)[0][0].data).abs().mean().cpu().numpy()
            fea_score =0

            with open(web_dir + '/' + result_name + '.txt', 'a') as f:
                print(i, dataTarget, dataPath, img_score, z_score, ssim_score, fea_score, center_score, file=f)

            print(i, dataTarget, dataPath, img_score, z_score, ssim_score, fea_score, center_score)

        print('0: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              '1: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              '2: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              '3: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              % (np.array(IMG_score_0).mean(), np.array(IMG_score_0).std(), np.array(Z_score_0).mean(), np.array(Z_score_0).std(), np.array(SSIM_score_0).mean(),
                 np.array(IMG_score_1).mean(), np.array(IMG_score_1).std(), np.array(Z_score_1).mean(), np.array(Z_score_1).std(), np.array(SSIM_score_1).mean(),
                 np.array(IMG_score_2).mean(), np.array(IMG_score_2).std(), np.array(Z_score_2).mean(), np.array(Z_score_2).std(), np.array(SSIM_score_2).mean(),
                 np.array(IMG_score_3).mean(), np.array(IMG_score_3).std(), np.array(Z_score_3).mean(), np.array(Z_score_3).std(), np.array(SSIM_score_3).mean()))

        with open(web_dir + '/1234.txt', 'a') as f:
            f.write('0: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              '1: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              '2: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
              '3: img - %.5f / %.5f, z - %.5f / %.5f, ssim - %.5f;\n'
            % (np.array(IMG_score_0).mean(), np.array(IMG_score_0).std(), np.array(Z_score_0).mean(), np.array(Z_score_0).std(), np.array(SSIM_score_0).mean(),
               np.array(IMG_score_1).mean(), np.array(IMG_score_1).std(), np.array(Z_score_1).mean(), np.array(Z_score_1).std(), np.array(SSIM_score_1).mean(),
               np.array(IMG_score_2).mean(), np.array(IMG_score_2).std(), np.array(Z_score_2).mean(), np.array(Z_score_2).std(), np.array(SSIM_score_2).mean(),
               np.array(IMG_score_3).mean(), np.array(IMG_score_3).std(), np.array(Z_score_3).mean(), np.array(Z_score_3).std(), np.array(SSIM_score_3).mean()))


        # print(Z_score)
        IMG_score = np.array(IMG_score)
        Z_score = np.array(Z_score)
        SSIM_score = np.array(SSIM_score)
        CENTER_score = np.array(CENTER_score)
        np.savez(web_dir  +'/' + save_name,
                 IMG_score=IMG_score, Z_score=Z_score, SSIM_score=SSIM_score, label=label, CENTER_score=CENTER_score)
    else:
        result = np.load(web_dir  +'/' + save_name)
        IMG_score = result['IMG_score']
        Z_score = result['Z_score']
        SSIM_score = result['SSIM_score']
        label = result['label']
        CENTER_score = result['CENTER_score']

    # all_score = 10*IMG_score + 100*Z_score #+ (1 - SSIM_score)
    if 'img' in save_name:
        all_score = IMG_score
    elif 'z' in save_name:
        all_score = Z_score

    label = np.array(label)
    label[label>0] = 1
    # label[label>0], label[label==0] = 0, 1

    f1, pre, rec, auc, acc, tn, fp, fn, tp, labelPred, thres = calc_score(label, all_score)
    sio.savemat(web_dir + '/label.mat', {'label':label, 'pred':labelPred, 'all_score': all_score,
                                         'thres':thres, 'fake_B':visuals['fake_B'], 'ad_B':visuals['ad_B']
                                         })
    
    print('Final score: F1-score is %.2f%%, precision is %.2f%%, recall is %.2f%%, accuracy is %.2f%%, auc is %.4f, '
               'tp is %d, tn is %d, fp is %d, fn is %d \n'
            % (f1*100, pre * 100, rec * 100, acc * 100, auc, tp, tn, fp, fn))

    with open(web_dir + '/1234.txt', 'a') as f:
       f.write('Final score: F1-score is %.2f%%, precision is %.2f%%, recall is %.2f%%, accuracy is %.2f%%, auc is %.4f, '
               'tp is %d, tn is %d, fp is %d, fn is %d \n'
            % (f1*100, pre * 100, rec * 100, acc * 100, auc, tp, tn, fp, fn))


