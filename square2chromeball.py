import os
import argparse 
import time
import torch
import skimage
import numpy as np
from tqdm.auto import tqdm


def create_argparser():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True ,help='directory that contain the square image') 
    parser.add_argument("--ball_dir", type=str, required=True ,help='directory to output hdr ball') #dataset name or directory 
    return parser


# create chromeball preview
def get_circle_mask(size=256):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(1, -1, size)
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    return mask


def cropped_ball(input_dir, output_dir):
    mask = get_circle_mask().numpy()
    files = os.listdir(input_dir)
    #os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(files):
      try:
        image = skimage.io.imread(os.path.join(input_dir, filename))
      except:
        continue
      image[mask == 0] = 0
      image = np.concatenate([image,  (mask*255)[...,None]], axis=2)
      image = image.astype(np.uint8)
      skimage.io.imsave(os.path.join(output_dir, filename), image)


if __name__ in '__main__':
    # load arguments
    start_time = time.time()
    args = create_argparser().parse_args()
    # make output directory if not exist
    os.makedirs(args.ball_dir, exist_ok=True)
    cropped_ball(args.input_dir, args.ball_dir)
    print(f"TOTAL TIME: {time.time()-start_time} sec.")