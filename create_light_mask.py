from glob import glob
import os
import argparse
import shutil
from scipy.special import sph_harm
import cv2
from PIL import Image, PngImagePlugin
import torch
import numpy as np
import json
import skimage
from tqdm import tqdm
from scipy.special import legendre, eval_legendre, lpmn, factorial
import scipy.constants as const
from skimage import data, exposure, img_as_float
from operator import itemgetter
from math import atan2
import time


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True ,help='directory that contain the presaved light masks') 
    parser.add_argument("--output_dir", type=str, required=True ,help='directory for saving refined light masks') 
    return parser


def recreate_light_mask(gaussian_centers, intensity, canva_size, kernel_size=0, radius=30, sigma=10, scale=1):
    '''
    Returns: a gaussian light mask image with smooth transition
    '''
    canva = np.zeros(canva_size, dtype=np.float32)
    summed_blur = np.zeros_like(canva, dtype=np.float32)
    for cntr in gaussian_centers:
        print(cntr)
        # apply multiple Gaussian_blurs and sum the results
        light_mask = cv2.circle(canva.copy(), (int(cntr[1]*scale), int(cntr[0]*scale)), radius, (1,), thickness=-1)
        light_mask = cv2.GaussianBlur(light_mask, (kernel_size, kernel_size), sigma).astype(np.float32)
        summed_blur += light_mask * 255 * np.mean(intensity)
    
    # normalize summed light spots
    light_mask = np.clip(summed_blur, 0, 255).astype(np.uint8)
    return light_mask


def light_mask_to_light_mask(light_mask_src, vis_dir):
    # read metadata
    img_path = os.path.dirname(light_mask_src)
    img_id = os.path.basename(light_mask_src)
    light = Image.open(light_mask_src)
    print(json.loads(light.info.get("light")))
    world_light = light.info.get("light")
    world_light = json.loads(world_light)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('light', json.dumps(world_light))

    msk_w, msk_h = light.size
    light_spot_scale = int(max(light.size)/256)
    light_mask = recreate_light_mask(gaussian_centers=world_light[:-1], 
                                     intensity=world_light[-1], 
                                     canva_size=(msk_h, msk_w), 
                                     radius=int(240 * msk_h/1024), 
                                     sigma=int(80 * msk_h/1024),
                                     scale=light_spot_scale)
    if max(light_mask.shape) != 1024:
        Image.fromarray(light_mask).resize((1024, 1024), Image.LANCZOS).save(os.path.join(vis_dir, img_id), pnginfo=metadata)
    else:
        Image.fromarray(light_mask).save(os.path.join(vis_dir, img_id), pnginfo=metadata)


if __name__ == "__main__":

    args = create_argparser().parse_args()
    search_dir = args.input_dir
    vis_dir = args.output_dir
    
    os.makedirs(vis_dir, exist_ok=True)
    print(search_dir, vis_dir)
    
    image_filename_list = glob(search_dir+'/*.png')
    images_path = [os.path.join(search_dir, file_path) for file_path in image_filename_list]
    print(len(images_path))

    for item in tqdm(images_path):
        
        light_mask_to_light_mask(item, vis_dir)
        p