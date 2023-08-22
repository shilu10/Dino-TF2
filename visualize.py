import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import torch
import tensorflow as tf 
from tensorflow import keras 
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from torch import nn
import torchvision
from copy import deepcopy
from vision_transformer import *


def get_arg_parser():
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--use_pretrained_weights', type=bool,
                        help='specify whether to use pretrained weights or saved models if pretrained weight is False', 
                        default='models/vit_base_patch16_224.dino')
    parser.add_argument('--model_path', type=str, 
                        help='if use_pretrained_weight is False, then need to specify the model_path', 
                        default='models/vit_base_patch16_224.dino')
    
    
    return parser
    

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def main(args):
    # build model
    if args.use_pretrained_weights:
        if 'base' in args.arch:
            model = vit_base(num_classes=0, patch_size=args.patch_size)
        elif 'small' in args.arch:
            model = vit_small(num_classes=0, patch_size=args.patch_size)
        elif 'tiny' in args.arch:
            model = vit_tiny(num_classes=0, patch_size=args.patch_size)
        else:
            print('specified architecture is not presented')
            sys.exit(1)
            
        model.load_weights(args.pretrained_weights)
    else:
        model.load_model(args.model_path)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)

    transform = pth_transforms.Compose([
            pth_transforms.Resize(args.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    img = transform(img)
    pytorch_img = deepcopy(img)
    img = img.permute(2, 1, 0)
    img = tf.Variable(img, dtype=tf.float32)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = tf.expand_dims(img, axis=0)

    pytorch_img = pytorch_img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[1] // args.patch_size
    h_featmap = img.shape[2] // args.patch_size

    attentions = model.get_last_selfattention(img)
    attentions = (attentions.numpy())

    attentions = torch.from_numpy(attentions)

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    for j in range(nh):
            fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
            
        
if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)