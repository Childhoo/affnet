# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:43:50 2019

@author: chenlin
"""


import matplotlib
matplotlib.use('Agg')
import os
import errno
import numpy as np
from PIL import Image
import sys
from copy import deepcopy
import argparse
import math
import torch.utils.data as data
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import gc
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import cv2
import copy
from Utils import L2Norm, cv2_scale
#from Utils import np_reshape64 as np_reshape
np_reshape = lambda x: np.reshape(x, (64, 64, 1))
from Utils import str2bool
from dataset import HPatchesDM,TripletPhotoTour, TotalDatasetsLoader
cv2_scale40 = lambda x: cv2.resize(x, dsize=(40, 40),
                                 interpolation=cv2.INTER_LINEAR)
from augmentation import get_random_norm_affine_LAFs,get_random_rotation_LAFs, get_random_shifts_LAFs
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches,normalizeLAFs
from pytorch_sift import SIFTNet
from HardNet import HardNet, L2Norm
#from Losses import loss_HardNetDetach, loss_HardNet
from Losses import loss_HardNet
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A,visualize_LAFs
import seaborn as sns
from Losses import distance_matrix_vector
from ReprojectionStuff import get_GT_correspondence_indexes

USE_CUDA = True

### Initialization
AffNetPix = AffNetFast(PS = 32)
#weightd_fname = '../../pretrained/AffNet.pth'
weightd_fname = 'logs/AffNetFast_lr005_10M_20ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNegC/checkpoint_18.pth'

checkpoint = torch.load(weightd_fname)
AffNetPix.load_state_dict(checkpoint['state_dict'])

AffNetPix.eval()

#load orientation net for more training process

OriNetPix = OriNetFast(PS=32)

def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    if USE_CUDA:
        var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape
def get_geometry_and_descriptors(img, det, desc, do_ori = True):
    with torch.no_grad():
        LAFs, resp = det(img,do_ori = do_ori)
        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = desc(patches)
    return LAFs, descriptors

#def get_geometry_and_descriptors_response(img, det, desc):
#    with torch.no_grad():
#        LAFs, resp = det(img, do_ori = True)
#        patches = detector.extract_patches_from_pyr(LAFs, PS = 32)
#        descriptors = descriptor(patches)
#    return LAFs, descriptors, resp

#def get_geometry_and_descriptors(img, det, desc, do_ori = True):
#    with torch.no_grad():
#        LAFs, resp = det(img,do_ori = do_ori)
#        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
#        descriptors = desc(patches)
#    return LAFs, descriptors

def test(epoch):
    torch.cuda.empty_cache()
    # switch to evaluate mode
    
    OriNetPix = OriNetFast(PS=32)
    weightd_fname_orinet = 'logs/OriNetFast_lr005_10M_20ep_aswap_ipidata_OriNet_6Brown_HardNet_0.005_10000000_HardNet/checkpoint_' + str(epoch) + '.pth'
    checkpoint_orinet = torch.load(weightd_fname_orinet)
    OriNetPix.load_state_dict(checkpoint_orinet['state_dict'])
    OriNetPix.eval()
    
    detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
                                          border = 5, num_Baum_iters = 1, 
                                          AffNet = AffNetPix, OriNet = OriNetPix)
    descriptor = HardNet()
    model_weights = 'HardNet++.pth'
    hncheckpoint = torch.load(model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.eval()
    if USE_CUDA:
        detector = detector.cuda()
        descriptor = descriptor.cuda()
    
    input_img_fname1 = 'test-graf/img1.png'#sys.argv[1]
    input_img_fname2 = 'test-graf/img6.png'#sys.argv[1]
    H_fname = 'test-graf/H1to6p'#sys.argv[1]
    output_img_fname = 'graf_match.png'#sys.argv[3]
    img1 = load_grayscale_var(input_img_fname1)
    img2 = load_grayscale_var(input_img_fname2)
    H = np.loadtxt(H_fname)    
    H1to2 = Variable(torch.from_numpy(H).float())
    SNN_threshold = 0.8
    with torch.no_grad():
        LAFs1, descriptors1 = get_geometry_and_descriptors(img1, detector, descriptor)
        torch.cuda.empty_cache()
        LAFs2, descriptors2 = get_geometry_and_descriptors(img2, detector, descriptor)
#        visualize_LAFs(img1.detach().cpu().numpy().squeeze(), LAFs1.detach().cpu().numpy().squeeze(), 'b', show = False, save_to = LOG_DIR + "/detections1_" + str(epoch) + '.png')
#        visualize_LAFs(img2.detach().cpu().numpy().squeeze(), LAFs2.detach().cpu().numpy().squeeze(), 'g', show = False, save_to = LOG_DIR + "/detection2_" + str(epoch) + '.png')
        dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
        min_dist, idxs_in_2 = torch.min(dist_matrix,1)
        dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
        min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
        mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
        tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False).cuda()[mask]
#        tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False)[mask]
        tent_matches_in_2 = idxs_in_2[mask]
        tent_matches_in_1 = tent_matches_in_1.long()
        tent_matches_in_2 = tent_matches_in_2.long()
        LAF1s_tent = LAFs1[tent_matches_in_1,:,:]
        LAF2s_tent = LAFs2[tent_matches_in_2,:,:]
        min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent,H1to2.cuda(), dist_threshold = 6) 
#        min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent,H1to2, dist_threshold = 6) 
        plain_indxs_in1 = plain_indxs_in1.long()
        inl_ratio = float(plain_indxs_in1.size(0)) / float(tent_matches_in_1.size(0))
        print('Test epoch', str(epoch)) 
        print('Test on graf1-6,', tent_matches_in_1.size(0), 'tentatives', plain_indxs_in1.size(0), 'true matches', str(inl_ratio)[:5], ' inl.ratio')
#        visualize_LAFs(img1.detach().cpu().numpy().squeeze(), LAF1s_tent[plain_indxs_in1.long(),:,:].detach().cpu().numpy().squeeze(), 'g', show = False, save_to = LOG_DIR + "/inliers1_" + str(epoch) + '.png')
#        visualize_LAFs(img2.detach().cpu().numpy().squeeze(), LAF2s_tent[idxs_in_2.long(),:,:].detach().cpu().numpy().squeeze(), 'g', show = False, save_to = LOG_DIR + "/inliers2_" + str(epoch) + '.png')
    return

for i in range(20):
    test(i)