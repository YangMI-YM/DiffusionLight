import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
import OpenEXR
import Imath
from scipy.special import sph_harm
from tqdm.auto import tqdm

from utils import hconcat_resize_min

# Constants
_WIDTH = _HEIGHT = 256
_RADIUS = 256

# SH basis functions (3-band, 9 coefficients)
def sh_basis(normal):
    x, y, z = normal
    return np.array([
        0.282095,                   # L_00
        0.488603 * y,              # L_1-1
        0.488603 * z,              # L_10
        0.488603 * x,              # L_11
        1.092548 * x * y,          # L_2-2
        1.092548 * y * z,          # L_2-1
        0.315392 * (3 * z**2 - 1), # L_20
        1.092548 * x * z,          # L_21
        0.546274 * (x**2 - y**2)   # L_22
    ])

# Convert environment map to SH coefficients
def envmap_to_sh(envmap, num_bands=3, num_coeff=9):
    H, W, _ = envmap.shape
    
    theta = np.linspace(0, np.pi, H)[:, None] # range (0, pi)
    phi = np.linspace(-np.pi, np.pi, W)[None, :] # range (-pi, pi)
    
    # Compute pixel direction vectors
    dirs = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta) * np.ones_like(phi)
    ], axis=-1)

    # Precompute sin(theta) * dOmega for integration weights
    dOmega = (2 * np.pi / W) * (np.pi / H) * np.sin(theta)

    sh = np.zeros((num_bands, num_coeff))
    for y in range(H):
        for x in range(W):
            normal = dirs[y, x]
            basis = sh_basis(normal)
            # L = envmap[y, x]  # RGB color
            # weight = dOmega[y, 0] # integration range over sphere
            for c in range(3): # RGB channel
                # color * basis * weight
                sh[c] += envmap[y, x, c] * basis * np.sin(theta[y, 0])
    sh /= (H * W)
    return sh

# Convert chrome ball to SH coeff
def chrome_ball_to_sh(ball_img, radius=256, num_coeffs=9):
    H, W, C = ball_img.shape
    cx, cy = W // 2, H // 2
    
    # Normalize coordinates to [-1, 1]
    ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    r2 = xs**2 + ys**2
    mask = r2 <= 1.0  # inside the ball

    # Compute normal directions from 2D coords
    z = np.sqrt(1.0 - r2, where=mask, out=np.zeros_like(r2))
    dirs = np.stack([xs, ys, z], axis=-1)  # [H, W, C]

    weights = z * mask  # projected solid angle approximation

    # Only process pixels inside the ball
    sh = np.zeros((C, num_coeffs), dtype=np.float64)

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            direction = dirs[y, x]
            basis = sh_basis(direction)
            for c in range(C):  # RGB/L*
                sh[c] += ball_img[y, x, c] * basis * weights[y, x]

    total_weight = np.sum(weights)
    sh /= total_weight  # normalize

    return sh

# Render gray ball using SH lighting
'''
def render_gray_ball(sh_coeffs, radius=_RADIUS, image_size=(_HEIGHT, _WIDTH)):
    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)
    center = (image_size[1] // 2, image_size[0] // 2)

    for y in range(image_size[0]):
        for x in range(image_size[1]):
            dx = (x - center[0]) / radius
            dy = (y - center[1]) / radius
            if dx**2 + dy**2 > 1:
                continue
            dz = np.sqrt(1 - dx**2 - dy**2)
            normal = np.array([dx, -dy, dz])
            basis = sh_basis(normal)
            if sh_coeffs.shape[0] == 3:
                color = np.zeros(3)
                for c in range(3):
                    color[c] = np.dot(sh_coeffs[c], basis)
                img[y, x] = np.clip(color, 0, 1)
            else:
                value = np.dot(sh_coeffs, basis)
                img[y, x] = value

    return (img * 255).astype(np.uint8)
'''
def render_gray_ball(sh, res=256):
    img = np.zeros((res, res))
    ys, xs = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing='ij')
    r2 = xs**2 + ys**2
    mask = r2 <= 1.0
    z = np.sqrt(1.0 - r2, where=mask, out=np.zeros_like(xs))

    for y in range(res):
        for x in range(res):
            if not mask[y, x]:
                continue
            normal = np.array([xs[y, x], ys[y, x], z[y, x]])
            basis = sh_basis(normal)
            value = np.sum([np.dot(sh[c], basis) for c in range(3)]) / 3.0  # gray = avg RGB
            img[y, x] = value

    return np.clip(img, 0, 1)

# Render a gray ball with a specified surface normal as the "up" direction
def render_light_probe(sh_coeffs, normal, resolution=64):
    img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    center = resolution // 2

    for y in range(resolution):
        for x in range(resolution):
            dx = (x - center) / center
            dy = (y - center) / center
            if dx**2 + dy**2 > 1:
                continue
            dz = np.sqrt(1 - dx**2 - dy**2)

            # Construct world space normal for this pixel (default sphere)
            local_normal = np.array([dx, -dy, dz])
            # Rotate to align with input normal
            n = align_normal(local_normal, np.array([0, 0, 1]), normal)
            basis = sh_basis(n)
            color = np.array([np.dot(sh_coeffs[c], basis) for c in range(3)])
            img[y, x] = np.clip(color, 0, 1)
    return img

# Rotate vector a to vector b
def align_normal(v, src, dst):
    v = np.array(v)
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    axis = np.cross(src, dst)
    angle = np.arccos(np.clip(np.dot(src, dst), -1, 1))
    if np.linalg.norm(axis) < 1e-6:
        return v  # No rotation needed
    axis = axis / np.linalg.norm(axis)
    return rotate_vector(v, axis, angle)

# Rodrigues' rotation formula
def rotate_vector(v, axis, angle):
    return (v * np.cos(angle) +
            np.cross(axis, v) * np.sin(angle) +
            axis * np.dot(axis, v) * (1 - np.cos(angle)))

# Create a grid of light probes for different surface directions
def generate_probe_grid(sh_coeffs, directions, resolution=64):
    n = len(directions)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid_img = np.ones((rows * resolution, cols * resolution, 3), dtype=np.float32)

    for i, dir in enumerate(directions):
        probe = render_light_probe(sh_coeffs, dir, resolution)
        row, col = divmod(i, cols)
        grid_img[row*resolution:(row+1)*resolution, col*resolution:(col+1)*resolution] = probe
    return grid_img

# Define directions for the probes (hemisphere sampling)
def uniform_hemisphere_samples(n=9):
    golden_angle = np.pi * (3 - np.sqrt(5))
    directions = []
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        if y >= 0:  # only upper hemisphere
            directions.append(np.array([x, y, z]))
    return directions


def estimate_global_intensity(sh_coeffs):
    '''
    # L_00 (first coefficient) represents average ambient light
    L₀₀ SH term approximates the average radiance over the sphere, and multiplying it by π gives total diffuse irradiance.
    '''
    return np.mean([c[0] for c in sh_coeffs]) * np.pi


def evaluate_sh_lighting(sh_coeffs, direction=np.array([0, 0, 1])):  # direction default: z-up
    '''
    probe lighting intensity from a specific direction by evaluating SH in that direction
    np.array([0, 0, 1]) # z-up
    np.array([1, 0, 0]) # 
    np.array([0, 1, 0]) # ...pending
    '''
    basis = sh_basis(direction)
    return np.array([np.dot(sh_coeffs[c], basis) for c in range(3)])


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


def lstar_to_luminance(Lstar):
    raise NotImplementedError
    

def perceived_light_intensity(Y, SH_coeff, epsilon=1e-6, min_Y_threshold=None):
    """
    Calculate a scalar that represents the overall perceived light intensity
    from a linear luminance (Y) image, considering only pixels above middle gray luminance.

    Returns a dictionary with different summary statistics.
    """
    if min_Y_threshold is None:
        Y = np.clip(Y, 0, 1)  # ensure values are in [0,1]

        mean_intensity = np.mean(Y)
        perc95 = np.percentile(Y, 95)
        geometric_mean = np.exp(np.mean(np.log(Y + epsilon)))

        return {
            "mean": mean_intensity,
            "perc95": perc95,
            "geom_mean": geometric_mean,
            "SH_coeff": SH_coeff
        }
    
    if isinstance(min_Y_threshold, float):
        mask = Y >= min_Y_threshold # 0.184
        # hemi sphere
        mask[128:, :] = False
    elif isinstance(min_Y_threshold, (np.ndarray, np.generic)):
        mask = min_Y_threshold.astype(bool)
    else:
        raise ValueError
    
    if not np.any(mask):
        return {
            "mean": 0.0,
            "perc95": 0.0,
            "geom_mean": 0.0,
            "coverage": 0.0,
            "SH_coeff": SH_coeff
        }
    Y_masked = Y[mask]
    light_area = np.sum(min_Y_threshold/255)
    #print(np.unique(Y_masked), np.unique(Y))
    mean_intensity = np.sum(Y_masked) / light_area #np.mean(Y_masked)
    #print(np.sum(Y_masked), light_area, mean_intensity)
    perc95 = np.percentile(Y_masked, 95)
    geometric_mean = np.exp(np.mean(np.log(Y_masked + epsilon)))
    coverage = np.sum(mask) / Y.size  # what % of pixels are "bright"

    return {
        "mean": mean_intensity,
        "perc95": perc95,
        "geom_mean": geometric_mean,
        "coverage": coverage,
        "SH_coeff": SH_coeff
    }



def image_to_lightness(image):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
        image = image.resize((512, 256))

    #if envmap.dtype != np.uint8:
    #    env_map_ = (np.clip(envmap, 0, 1) * 255).astype(np.uint8)
    #else:
    #    env_map_ = cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR)
    
    img_np = np.clip(np.asarray(image), 0, 1) #np.asarray(image) / 255.0  # Normalize to [0, 1]

    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]

    Y = rgb_to_luminance(r, g, b)
    
    L_star = luminance_to_lstar(Y)

    return L_star, Y


def light_spots_mask(lumi_img, mid_gray=50):
    lumi_mask = np.zeros(lumi_img.shape[:2], dtype=np.uint8)
    lumi_mid_thres = lumi_img > mid_gray
    meme = np.unique(lumi_img[lumi_mid_thres])
    while len(np.unique(lumi_mid_thres)) < 2:
        mid_gray -= 12
        lumi_mid_thres = lumi_img > mid_gray
    
    lumi_mid_thres = (lumi_mid_thres * 255).astype(np.uint8)
    #print(np.count_nonzero(lumi_mid_thres), np.max(lumi_img), meme, "lumi masked")
    light_spot_cnts, _ = cv2.findContours(lumi_mid_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # find the brightest light spot
    light_spot_energy, highlight = [], None
    for idx in range(len(light_spot_cnts)):
        cimg = np.zeros_like(lumi_img)
        cv2.drawContours(cimg, light_spot_cnts, idx, color=255, thickness=-1)
        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        #lst_intensities.append(img[pts[0], pts[1]])
        light_spot_energy.append(np.mean(lumi_img[pts[0], pts[1]]))
    highlight = max(light_spot_energy)
    #print("light spots energy", light_spot_energy, highlight)
    for ls_cnt, ls_energy in zip(light_spot_cnts, light_spot_energy):
        M = cv2.moments(ls_cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            area = cv2.contourArea(ls_cnt)
            if cy < 128 and ls_energy > 0.9*highlight:
                cv2.drawContours(lumi_mask, [ls_cnt], -1, 255, -1)
            elif area > 4e2 and ls_energy > 0.95*highlight: # percentile90 ?
                cv2.drawContours(lumi_mask, [ls_cnt], -1, 255, -1)
    lumi_mask[lumi_mid_thres==0] = 0 # my patch
    #print(np.unique(lumi_img[lumi_mask.astype(bool)]))
    return lumi_mask


# Main function
def read_env_map(input_dir, output_dir):
    '''
    chrome_ball_list = glob(os.path.join(input_dir, "*ev-50.png"))
    for chrome_ball in tqdm(chrome_ball_list):
        img_id, img_format = os.path.basename(chrome_ball).split("_ev-50")
        # ---------- Load chrome ball image (RGBA) ----------
        chrome_rgba = iio.imread(chrome_ball)  
        chrome_rgb = chrome_rgba[..., :3] / 255.0

        # Get SH coefficients 
        sh_coeffs = chrome_ball_to_sh(chrome_rgb)

        # Render gray ball
        rendered_rgb = render_gray_ball(sh_coeffs)

        # ---------- Convert to Lab and compute SH from L* only ----------
        chrome_lab = cv2.cvtColor((chrome_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
        chrome_l = chrome_lab[..., 0] / 255.0  # Normalize L* to [0, 1]
        sh_coeffs_l = chrome_ball_to_sh(chrome_l[:,:, None])
        rendered_l = render_gray_ball(sh_coeffs_l)
    '''
    # Load environment map (HDR)
    hdr_filename_list = glob(os.path.join(input_dir, "*.exr"))
    for hdr_envmap in tqdm(hdr_filename_list):
        hdr_id = os.path.basename(hdr_envmap).strip(".exr")
        exr_file = OpenEXR.InputFile(hdr_envmap)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Read each channel
        r = np.frombuffer(exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size[1], size[0])
        g = np.frombuffer(exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size[1], size[0])
        b = np.frombuffer(exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size[1], size[0])

        envmap = np.stack([r, g, b], axis=-1)
        # in range [0, 1] for float32, or [0, 255] for uint8
        if envmap.dtype != np.uint8:
            env_map_ = (np.clip(envmap, 0, 1) * 255).astype(np.uint8)
        else:
            env_map_ = cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR)

        chrome_rgba = iio.imread(glob(hdr_envmap.replace("hdr", "ball").strip(".exr")+"_ev-00.png")[0])  

        sh_coeffs = envmap_to_sh(envmap, )
        # Render gray ball
        rendered = render_gray_ball(sh_coeffs)

        # tiled imgs
        tiled = hconcat_resize_min([chrome_rgba[..., :3], 
                                    cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                                    env_map_]) 
        cv2.imwrite(os.path.join(output_dir, f"{hdr_id}_tiled.png"), cv2.cvtColor(tiled, cv2.COLOR_RGB2BGR))

        # light probes from all directions 
        dirs = uniform_hemisphere_samples(n=16)
        probe_grid = generate_probe_grid(sh_coeffs, dirs, resolution=256)

        # Save result
        cv2.imwrite(os.path.join(output_dir, f"{hdr_id}_probe.png"), 
                    cv2.cvtColor((np.clip(probe_grid, 0, 1) * 255).astype(np.uint8), 
                                cv2.COLOR_RGB2BGR))
        #plt.imsave(probe_grid, os.path.join(output_dir, f"{hdr_id}_probe.png"))

        # TODO 
        global_intensity = estimate_global_intensity(sh_coeffs)
        top_light = evaluate_sh_lighting(sh_coeffs, np.array([0, 0, 1]))
        avg_irradiance = np.mean(rendered[np.any(rendered > 0, axis=-1)], axis=0) / 255.0
        
        print("🔆 Global Light Intensity (approx):", global_intensity, hdr_id)
        #print("🌅 Top Direction Intensity (RGB):", top_light)
        #print("🎯 Average Irradiance on Sphere (RGB):", avg_irradiance)


def visualize_lightness(L_star, Y, SH_coeff, mid_gray_Lstar=80, output_file="lightness_summary.png"):
    """middle gray: L* = 50 is the equivalent of Y = 18.4, or in other words an 18% grey card"""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    ax1, ax2, ax3, ax4, ax5 = axes

    # L* map
    ax1.imshow(L_star, cmap='gray', vmin=0, vmax=100)
    ax1.set_title("Perceived Lightness (L*)")
    ax1.axis('off')

    # Light region mask
    middle_gray_Y = ((mid_gray_Lstar + 16) / 116) ** 3  # ~0.184
    mask_above_middle_gray = Y >= middle_gray_Y
    #mask_above_middle_gray[128:, :] = False
    ax2.imshow(mask_above_middle_gray, cmap='gray')
    ax2.set_title(f"Mask: Y ≥ {middle_gray_Y} (Light Sources)")
    ax2.axis('off')

    # Mask
    mask_above_middle_gray = light_spots_mask(L_star, mid_gray_Lstar)
    #mask_above_middle_gray = L_star > mid_gray_Lstar # middle_gray
    #mask_above_middle_gray[128:, :] = False
    ax3.imshow(mask_above_middle_gray, cmap='gray')
    ax3.set_title(f"Mask: L* > {mid_gray_Lstar}")
    ax3.axis('off')

    # Histogram
    ax4.hist(L_star.ravel(), bins=50, range=(0, 100), color='slateblue', edgecolor='black')
    ax4.set_title("L* Histogram")
    ax4.set_xlabel("L* Value")
    ax4.set_ylabel("Pixel Count")

    # Bar plot for perceived intensity
    intensity = perceived_light_intensity(Y, SH_coeff, min_Y_threshold=mask_above_middle_gray)
    bars = ax5.bar(intensity.keys(), intensity.values(), color='goldenrod')
    ax5.set_ylim(0, 1.20)
    ax5.set_title("Perceived Intensity (Y summary)")
    ax5.set_ylabel("Relative Value")

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f"{height:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    #plt.show()

    print(f"Visualization saved to '{output_file}'")
    return intensity['mean']



def lumi_ch_extract(input_dir, output_dir):
    # Load environment map (HDR)
    intensities = {}
    hdr_filename_list = glob(os.path.join(input_dir, "*.exr"))[243:]
    for hdr_envmap in tqdm(hdr_filename_list):
        hdr_id = os.path.basename(hdr_envmap).strip(".exr")
        #if "boy__smells380452105_6672384816186956_3831334335444567758_n" not in hdr_id:
        #    continue
        exr_file = OpenEXR.InputFile(hdr_envmap)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Read each channel
        r = np.frombuffer(exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size[1], size[0])
        g = np.frombuffer(exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size[1], size[0])
        b = np.frombuffer(exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(size[1], size[0])

        envmap = np.stack([r, g, b], axis=-1)
        # in range [0, 1] for float32, or [0, 255] for uint8
        if envmap.dtype != np.uint8:
            env_map_ = (np.clip(envmap, 0, 1) * 255).astype(np.uint8)
        else:
            env_map_ = cv2.cvtColor(envmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{hdr_id}_envmap.png"), env_map_)
        #envmap_lab = cv2.cvtColor((envmap * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
        env_map_lstar, env_map_Y = image_to_lightness(envmap)
        envmap_lab = np.stack((env_map_Y,)*3, axis=-1)
        sh_coeffs = envmap_to_sh(envmap_lab)
        mean_light_energy = visualize_lightness(env_map_lstar, env_map_Y, sh_coeffs[0][0], output_file=os.path.join(output_dir, f"{hdr_id}_L*.png"))
        intensities[hdr_id] = [mean_light_energy, estimate_global_intensity(envmap_to_sh(envmap)), sh_coeffs[0][0]]
        continue

        chrome_rgba = iio.imread(glob(hdr_envmap.replace("hdr", "ball").strip(".exr")+"_ev-00.png")[0])  

        print(sh_coeffs)
        print(f"L* stats:\n  Min: {np.min(env_map_lstar):.2f}\n  Max: {np.max(env_map_lstar):.2f}\n  Mean: {np.mean(env_map_lstar):.2f}")
        
        # Render gray ball
        rendered = render_gray_ball(sh_coeffs)

        # tiled imgs
        tiled = hconcat_resize_min([chrome_rgba[..., :3], 
                                    cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                                    env_map_]) 
        cv2.imwrite(os.path.join(output_dir, f"{hdr_id}_tiled_L.png"), cv2.cvtColor(tiled, cv2.COLOR_RGB2BGR))

        # light probes from all directions 
        dirs = uniform_hemisphere_samples(n=16)
        probe_grid = generate_probe_grid(sh_coeffs, dirs, resolution=256)

        # Save result
        cv2.imwrite(os.path.join(output_dir, f"{hdr_id}_probe_L.png"), 
                    cv2.cvtColor((np.clip(probe_grid, 0, 1) * 255).astype(np.uint8), 
                                cv2.COLOR_RGB2BGR))
        #plt.imsave(probe_grid, os.path.join(output_dir, f"{hdr_id}_probe.png"))

        # TODO 
        global_intensity = estimate_global_intensity(sh_coeffs)
        top_light = evaluate_sh_lighting(sh_coeffs, np.array([0, 0, 1]))
        avg_irradiance = np.mean(rendered[np.any(rendered > 0, axis=-1)], axis=0) / 255.0
        
        print("🔆 Global Light Intensity (approx):", global_intensity, hdr_id)
        #print("🌅 Top Direction Intensity (RGB):", top_light)
        #print("🎯 Average Irradiance on Sphere (RGB):", avg_irradiance)
    
    print("sorted by intensity", dict(sorted(intensities.items(), key=lambda item: item[1][0])))

if __name__ == "__main__":
    input_dir = "/home/yangmi/s3data-3/beauty-lvm/v2/light/768/batch_0/hdr"
    output_dir = "/home/yangmi/s3data-3/beauty-lvm/v2/light/768/batch_0/intensity"
    os.makedirs(output_dir, exist_ok=True)
    #read_env_map(input_dir, output_dir)
    lumi_ch_extract(input_dir, output_dir)