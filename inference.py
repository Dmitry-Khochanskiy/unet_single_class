#!/usr/bin/env python
# coding: utf-8


import time
import os
import numpy as np
import torch
import torchvision.transforms as tv
import torchvision
import logging
import matplotlib.pyplot as plt
import pandas as pd
import glob
from torchvision.utils import make_grid, save_image
from PIL import Image
from torchvision.transforms import Compose
from UNet import UNet

'''Inference workflow
 model = load_model(model_path) ->
 show_img_with_pred(image_path, model, show_results=1) | batch_prediction(images_folder_path, model, save_path=folder) '''


mylogs = logging.getLogger()
mylogs.setLevel(logging.INFO)
file = logging.FileHandler("inference.log", mode='w')
file.setLevel(logging.INFO)
fileformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s",datefmt="%H:%M:%S")
file.setFormatter(fileformat)
mylogs.addHandler(file)
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
mylogs.addHandler(stream)
mylogs.info("Inference")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mylogs.info(f"Device is: {str(device)}")

def load_model(model_path):
    ''' Loads a model and its state dict. Accepts path to the model, returns the model. Default cnn type is resnet18 '''
    saved_model = torch.load(f'{model_path}')
    model_name  = saved_model['model_name']
    model_state_dict = saved_model['model_state_dict']
    size, n_channels, n_classes =  saved_model['parameters']
    mylogs.info("Model: " + model_name)

    model = UNet(n_channels, n_classes)
    model.load_state_dict(model_state_dict)
    return model, size, n_channels

def image_loader(image_path, size,n_channels):
    '''loads image,transforms it and returns  tensor'''
    img = Image.open(image_path)
    img = img.resize((size[0], size[1]))
    img = transform(n_channels)(img)
    #As torch models expects a batch, this image should be turned into a batch with 1 image
    img  = img.unsqueeze(0)
    return img.cpu()

def show_img_with_seg(image_path, model, size, n_channels, show_results=0):
    ''' Predicts one image, returns masked imgs and masks '''
    img = image_loader(image_path,size,n_channels)
    img = img.to(device)
    model.to(device)
    with torch.no_grad():
        prediction_mask = model(img)
        prediction_mask[prediction_mask > 0] = 1
        prediction_mask[prediction_mask <= 0] = 0
        masked_img = img + prediction_mask.int().float()

    if show_results:
        img = img.cpu()
        masked_img = masked_img.cpu()
        prediction_mask = prediction_mask.cpu()
        imgs = (img, masked_img, prediction_mask)
        n_row = 1
        n_col = 3
        _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
        axs = axs.flatten()
        titles = [' Original Image', 'Image with mask', 'Mask']
        for title, img, ax in zip(titles, imgs, axs):
            ax.imshow(make_grid(img, 4).permute(1,2,0))
            ax.set_title(title)
        plt.show()
    else:
        return masked_img, prediction_mask

def transform(n_channels):
    return tv.Compose([
        tv.ToTensor(),
            tv.Normalize((0), (1))
        ])

def save_img_or_masks(img, image_path,image_save_path, prefix = None):
    save_image(img, image_save_path + "\\" + os.path.basename(image_path).split('.')[0] + prefix + ".png")

def batch_prediction(images_folder_path,model,size, n_channels, img_save_path, mask_save_path=None):
    ''' Make an inference on the images in the folder, returns a csv with results'''
    imgs_path_list = glob.glob(f'{images_folder_path}\*')
    mylogs.info(f"Inference started. \n{len(imgs_path_list)} images to segment")
    if not mask_save_path:
        mask_save_path = img_save_path
    start_time = time.time()
    for image_path in imgs_path_list:
        seg_img, mask = show_img_with_seg(image_path, model,size, n_channels, show_results=None)
        save_img_or_masks(seg_img, image_path, img_save_path, prefix = "_segmented_")
        save_img_or_masks(mask,image_path, mask_save_path, prefix = "_mask_")
    end_time = time.time()
    total_time = int(end_time - start_time)
    mylogs.info(f'Inference finished. Elapsed time: {total_time }s')
