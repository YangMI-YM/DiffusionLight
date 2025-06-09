import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2) # this works
from skimage import data, exposure
from skimage.color import rgb2lab, lab2rgb
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float
from scipy.ndimage import gaussian_filter
from scipy.special import sph_harm
from tqdm.auto import tqdm
import json
import argparse


def srgb_to_linear(channel):
    return np.where(
        channel <= 0.04045,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4
    )


def rgb_to_luminance(r, g, b):
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def luminance_to_lstar(Y):
    epsilon = 216 / 24389
    kappa = 24389 / 27
    return np.where(
        Y <= epsilon,
        Y * kappa,
        116 * (Y ** (1 / 3)) - 16
    )


def image_to_lightness(image):
    
    img_np = np.asarray(image) / 255.0  # Normalize to [0, 1]

    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]

    Y = rgb_to_luminance(r, g, b)
    
    L_star = luminance_to_lstar(Y)

    return L_star, Y


# --- 1. Local Brightness Normalization ---
def local_brightness_normalization(L, Lstar, patch_size, threshold):
    h, w = L.shape
    selected_means = []
    vis = np.zeros_like(L)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = L[y:y+patch_size, x:x+patch_size]
            mean_val = np.mean(patch)
            if mean_val > threshold:
                selected_means.append(mean_val)
                vis[y:y+patch_size, x:x+patch_size] = Lstar[y:y+patch_size, x:x+patch_size] #mean_val
    if len(selected_means) == 0:
        return 0.0, vis
    brightness = np.mean(selected_means)
    print(f"brightness: {brightness}")
    return brightness, vis


# --- 2. CLAHE Enhancement ---
def apply_CLAHE(L):
    L_scaled = np.uint8(255 * L)  # Scale L to [0,255] for OpenCV
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L_scaled)
    return L_eq/255, L_eq * (100 / 255)  # Scale to [0,100]


# --- 3. Simple Retinex Approximation (SSR) ---
def single_scale_retinex(img, sigma=30):
    # single_scale_retinex(L_channel)
    blurred = gaussian_filter(img, sigma)
    retinex = np.log1p(img) - np.log1p(blurred + 1e-6)
    retinex_norm = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 100 # Normalize to [0,100]
    return retinex, retinex_norm


def find_brightness(input_dir, output_dir, patch_size=32, middle_gray_threshold=50):
    ''' luminance (Y) & Log-Average Luminance
    luminance = 0.2126 * hdr_image[..., 2] + 0.7152 * hdr_image[..., 1] + 0.0722 * hdr_image[..., 0]
    global_brightness = np.mean(luminance)
    epsilon = 1e-4
    log_avg_lum = np.exp(np.mean(np.log(epsilon + luminance))) '''

    middle_gray_Y = ((middle_gray_threshold + 16) / 116) ** 3  # ~0.184

    brightness = {}
    img_filename_list = glob(os.path.join(input_dir, "*.*"))#[:10]
    for hdr_img in tqdm(img_filename_list):
        img_id = '.'.join(os.path.basename(hdr_img).split('.')[:-1])
        img_rgb = Image.open(hdr_img).resize((1024, 1024))
        img_Lstar, img_Y = image_to_lightness(img_rgb)

        # local norm
        local_brightness, local_vis = local_brightness_normalization(img_Y, img_Lstar, patch_size, middle_gray_Y)
        # CLAHE
        clahe_Y, clahe_Lstar = apply_CLAHE(img_Y) # img_Y ~ [0, 1]
        clahe_local_brightness, clahe_local_vis = local_brightness_normalization(clahe_Y, clahe_Lstar, patch_size, middle_gray_Y)
        # Retinex
        #retinex_L, retinex_norm = single_scale_retinex(img_Y)
        # mask-based
        #shadow_mask = iio.imread(os.path.join(output_dir.replace("brightness", "shadow_SAM2Adapter"), img_id+".jpeg"))
        #masked_L = np.copy(img_Y)
        #masked_L[shadow_mask] = np.nan
        #shadow_masked_brightness = np.nanmean(masked_L)

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(img_Lstar, cmap='gray', vmin=0, vmax=100)
        axs[0, 1].set_title("L* Channel")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(local_vis, cmap='gray', vmin=0, vmax=100)
        axs[0, 2].set_title(f"Local Normalization (L* > 50): {local_brightness:.2f}")
        axs[0, 2].axis('off')

        axs[1, 0].hist(clahe_Lstar.ravel(), bins=50, range=(0, 100), color='slateblue', edgecolor='black')
        axs[1, 0].set_title("CLAHE L* Histogram")
        axs[1, 0].set_xlabel("L* Value")
        axs[1, 0].set_ylabel("Pixel Count")
        #axs[1, 0].axis('off')

        axs[1, 1].imshow(clahe_Lstar, cmap='gray', vmin=0, vmax=100)
        axs[1, 1].set_title("CLAHE Enhanced L*")
        axs[1, 1].axis('off')

        axs[1, 2].imshow(clahe_local_vis, cmap='gray')
        axs[1, 2].set_title(f"CLAHE Enhanced Local Normalization (L* > 50): {clahe_local_brightness:.2f}")
        axs[1, 2].axis('off')

        plt.suptitle("Comparison of Brightness Enhancement Techniques", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, img_id+".png"), dpi=150)
        plt.close('all')

        brightness[img_id] = [local_brightness, clahe_local_brightness]

    #print("sorted by brightness", dict(sorted(brightness.items(), key=lambda item: np.mean(item[1]))))
    
    json_obj = json.dumps(brightness, indent=2)
    with open(os.path.join(output_dir, "brightness.json"), "w") as fo:
        fo.write(json_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True) # "/home/yangmi/s3data-3/beauty-lvm/v2/light"
    args = parser.parse_args()
    input_dir = args.input_dir
    resolution, batch_name = input_dir.split("/")[-3:-1]
    
    output_dir = os.path.join(args.output_dir, resolution, batch_name, "brightness")
    #os.makedirs(output_dir, exist_ok=True)
    #find_brightness(input_dir, output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    find_brightness(input_dir, args.output_dir)