# /************************************************************************
# *
# * ADOBE CONFIDENTIAL
# * ___________________
# *
# * Copyright 2021 Adobe
# * All Rights Reserved.
# *
# * NOTICE: All information contained herein is, and remains
# * the property of Adobe and its suppliers, if any. The intellectual
# * and technical concepts contained herein are proprietary to Adobe
# * and its suppliers and are protected by all applicable intellectual
# * property laws, including trade secret and copyright laws.
# * Dissemination of this information or reproduction of this material
# * is strictly forbidden unless prior written permission is obtained
# * from Adobe.
# *************************************************************************
# */

import os
import glob
import argparse

import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ckpt_path', default='./ckpt/maskdepth_model.pkl', help='Checkpoint path')
parser.add_argument('--test_input_rgb_dir', default='./images/input/rgb', help='RGB input dir')
parser.add_argument('--test_input_depth_dir', default='./images/input/depth', help='Depth input dir')
parser.add_argument('--test_input_mask_dir', default='./images/input/mask', help='Mask input dir')
parser.add_argument('--test_output_dir', default='./images/output', help='Output dir')

parser.add_argument('--input_size', default=512, type=int, help='Input size (multiples of 32)')
parser.add_argument('--hires', action='store_true', default=False, help='High resolution results')
parser.add_argument('--colormap', default='inferno', help='Colormap for plt.imsave() - inferno, gray, etc')
args = parser.parse_args()


def scale_torch(img, is_rgb=False):
    """
    Scale the image and output it in torch.tensor.
    :param img: rgb is in shape [H, W, C], depth/disp or mask is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if not is_rgb:
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))])
        img = transform(img)
    return img


if __name__ == '__main__':
    # set configs
    depth_scale = 10
    
    # read input images
    test_rgb_list = sorted(sum([glob.glob(os.path.join(args.test_input_rgb_dir, ext)) for ext in ['*.png', '*.jpg', '*.jpeg']], []))
    test_depth_list = sorted(sum([glob.glob(os.path.join(args.test_input_depth_dir, ext)) for ext in ['*.png', '*.jpg', '*.jpeg']], []))
    test_mask_list = sorted(sum([glob.glob(os.path.join(args.test_input_mask_dir, ext)) for ext in ['*.png', '*.jpg', '*.jpeg']], []))
    
    assert len(test_rgb_list)==len(test_depth_list), "Number of depth maps and rgb images should be the same!!!"
    assert len(test_mask_list)==len(test_depth_list), "Number of depth maps and masks should be the same!!!"
    test_data_size = len(test_depth_list)
    
    if not os.path.exists(args.test_output_dir):
        os.makedirs(args.test_output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load model
    model = pickle.load(open(args.ckpt_path, 'rb'))
    model.eval().to(device).requires_grad_(False).to(device)
    print('Model loaded from:', args.ckpt_path)

    for ind in range(test_data_size):
        print('[%04d/%04d] Depth: %s, Mask: %s' % (ind+1, test_data_size, test_depth_list[ind], test_mask_list[ind]))
        
        # read inputs
        rgb = cv2.imread(test_rgb_list[ind])  # [H, W, 3]
        rgb = rgb[:, :, ::-1]
        
        depth = cv2.imread(test_depth_list[ind], -1)  # [H, W]
        depth = depth / (depth.max()+1e-8)
        depth = 1 - depth
        
        mask = cv2.imread(test_mask_list[ind], -1)  # [H, W]
        if len(mask.shape) == 3:  # if 3 channels are given for mask
            mask = np.amax(mask, axis=2)
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.
        
        h, w = depth.shape
        if h > args.input_size or w > args.input_size:
            output_size = (w, h)
            input_size = (args.input_size, args.input_size)
        else:
            output_size = (w - np.mod(w, 32), h - np.mod(h, 32))
            input_size = output_size
        
        # resize
        rgb_resize = cv2.resize(rgb, input_size, interpolation=cv2.INTER_AREA)
        depth_resize = cv2.resize(depth, input_size, interpolation=cv2.INTER_AREA)
        depth_resize = depth_resize / (depth_resize.max() + 1e-8) * depth_scale
        mask_resize = cv2.resize(mask, input_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_AREA)
        
        # generate data dict
        rgb_torch = torch.unsqueeze(scale_torch(rgb_resize, is_rgb=True), dim=0).to(device)
        depth_torch = torch.unsqueeze(scale_torch(depth_resize), dim=0).to(device)
        mask_torch = torch.unsqueeze(scale_torch(mask_resize), dim=0).to(device)
        
        data = {'input_rgb': rgb_torch, 'input_depth': depth_torch, 'mask': mask_torch}
        
        # inference
        pred, pred_fg, pred_bg = model(data)
        
        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy()
            pred_fg = pred_fg.cpu().detach().numpy()
            pred_bg = pred_bg.cpu().detach().numpy()
            
        # post-process
        pred = np.squeeze(np.clip(pred / depth_scale, 0, 1))
        pred_fg = np.squeeze(np.clip(pred_fg / depth_scale, 0, 1))
        pred_bg = np.squeeze(np.clip(pred_bg / depth_scale, 0, 1))
        
        if args.hires:
            # resize and composite with hires mask
            pred_fg = cv2.resize(pred_fg, output_size, interpolation=cv2.INTER_LINEAR)
            pred_bg = cv2.resize(pred_bg, output_size, interpolation=cv2.INTER_LINEAR)
            pred = pred_fg * mask + pred_bg * (1 - mask)
        else:
            pred = cv2.resize(pred, output_size, interpolation=cv2.INTER_LINEAR)
            
        # disparity ordering
        pred = 1 - pred

        # save result images
        filename = os.path.join(args.test_output_dir, os.path.basename(test_depth_list[ind]))
        plt.imsave(filename[:-4]+'_input.png', 1 - depth, cmap=args.colormap, vmin=0, vmax=1)
        plt.imsave(filename[:-4]+'_output.png', pred, cmap=args.colormap, vmin=0, vmax=1)
        
        print('Image saved to: {}'.format(filename))
