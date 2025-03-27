###Utils
from glob import glob
import os
from scipy.special import sph_harm
import cv2
from PIL import Image
import torch
import numpy as np
import skimage
import os
from scipy.special import legendre, eval_legendre, lpmn, factorial
import scipy.constants as const
from skimage import data, exposure, img_as_float
from operator import itemgetter
from math import atan2, pi


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
    os.makedirs(output_dir, exist_ok=True)
    for filename in files:
        try:
            image = skimage.io.imread(os.path.join(input_dir, filename))
        except:
            continue
        image[mask == 0] = 0
        image = np.concatenate([image,  (mask*255)[...,None]], axis=2)
        image = image.astype(np.uint8)
        skimage.io.imsave(os.path.join(output_dir, filename), image)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_LANCZOS4):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_LANCZOS4):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    im_list_resize = [np.stack([im, im, im], axis=-1) if len(im.shape)==2 else im for im in im_list_resize]
    
    return cv2.hconcat(im_list_resize)


def closest_point(point, array):
    diff = array - point
    distance = np.einsum('ij,ij->i', diff, diff)
    return np.argmin(distance), distance


def contour_dist(cnt_cur, cnt_ref):
    # initialize some variables
    min_dist = np.inf
    chosen_point_c2 = None
    chosen_point_c1 = None

    # iterate through each point in contour c1
    for point in cnt_cur:
        t = point[0][0], point[0][1]
        index, dist = closest_point(t, cnt_ref[:,0])
        if dist[index] < min_dist :
            min_dist = dist[index]
            chosen_point_c2 = cnt_ref[index]
            chosen_point_c1 = t
    return min_dist, chosen_point_c1, chosen_point_c2


def equal_blend(list_images, list_ev): # Blend images equally.
    '''
    opencv gamma correction lookup 
    0 => 0
    1 => 2
    2 => 4
    3 => 6
    4 => 7
    5 => 9
    6 => 11
    7 => 12
    8 => 14
    9 => 15
    10 => 17

    skimage gamma correction lookup

    '''
    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img, ev in zip(list_images, list_ev):
        img = exposure.adjust_gamma(img_as_float(img), ev) # gamma>1 hist shift towards left
        output = output + img * equal_fraction

    output = (output * 255).astype(np.uint8)
    return output


def pil_square_crop_image(image, desired_size = (512, 512), interpolation=Image.LANCZOS):
    """
    Make top-bottom border
    """
    # Don't resize if already desired size (Avoid aliasing problem)
    if image.size == desired_size:
        return image

    # Center crop
    width, height = image.size
    new_width, new_height = min(width, height), min(width, height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))

    # Calculate the scale factor
    scale_factor = min(desired_size[0] / image.width, desired_size[1] / image.height)

    # Resize the image
    resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), interpolation)

    # Create a new blank image with the desired size and black border
    new_image = Image.new("RGB", desired_size, color=(0, 0, 0))

    # Paste the resized image onto the new image, centered
    new_image.paste(resized_image, ((desired_size[0] - resized_image.width) // 2, (desired_size[1] - resized_image.height) // 2))

    return new_image


def transform2Dpos2Spherical3D(pixel_pos, dim=256, light_type='point'):
    '''Ref: https://playground.babylonjs.com/#2GPSRC#2
        Step 1: (x_p, y_p) pixel coordinate -> (theta, phi) spherical coordinates
        Step 2: (theta, phi) -> (x, y, z) world coordinate
    '''
    #print(f"points in (x, Y) coordinate {pixel_pos}.")

    world_pos = []
    if len(pixel_pos) == 0:
        return world_pos

    if light_type == 'point':
        radius = dim
    elif light_type == 'directional':
        radius = dim * 10.0
    else:
        raise NotImplementedError

    for pos in pixel_pos:
        # scale to [0, 1]
        # theta is in range [0, pi],
        #theta = pos[0] / (dim / 360) - 180
        theta = atan2(pos[1], pos[0]) # azimuthal angle <pos, positive-x-axis> in xy-plane
        # phi is in range [-pi/2, pi/2]
        #phi = - (pos[1] / (dim / 180) - 90)
        phi = 90 # polar angle is always 90 for a flat circle in xy-plane

        #print(f"theta_phi range {np.rad2deg(theta)} \in [0, \pi].")

        phi = np.deg2rad(phi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        world_pos.append((x, y, z))

    return world_pos


def detect_contours(img):
    if img.shape[-1] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    equalized_img = cv2.equalizeHist(gray_img)
    blurred_img = cv2.GaussianBlur(equalized_img, (9, 9), 0)
    edges = cv2.Canny(blurred_img, 90, 180)
    _, thresh = cv2.threshold(edges, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    return contours


def calculate_contour_distance(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return max(abs(c_x1 - c_x2) - (w1 + w2)/2, abs(c_y1 - c_y2) - (h1 + h2)/2)


def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)


def agglomerative_cluster(contours, threshold_distance=40.0):
    if len(contours) == 1 and cv2.contourArea(contours[0]) > 10: #
        return contours, [] # must be
    # split into point_light & directional_light categories
    pt_light_cnts = []
    dir_light_cnts = []
    for cnt in contours:
        dist = cv2.pointPolygonTest(cnt, (128,128), True) # signed distance between the point and the nearest contour edge
        #print("distance to sphere center", np.abs(dist))
        if np.abs(dist) < 108: # 128-15
            pt_light_cnts.append(cnt)
        else:
            dir_light_cnts.append(cnt)
    # 6*6 is nearly invisible to me
    current_contours = list(filter(lambda x: cv2.contourArea(x)>36, pt_light_cnts))
    #print(f"from agglomerative cnt cates {len(pt_light_cnts)}, {len(current_contours)}")

    if len(current_contours) == 0 and len(pt_light_cnts) > 2:
        # returns the brightest
        return pt_light_cnts, dir_light_cnts

    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours)-1):

            for y in range(x+1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else:
            break

    return current_contours, dir_light_cnts


def cartesian_to_spherical(cartesian_coordinates):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coordinates: A NumPy array of shape [..., 3], where each row
        represents a Cartesian coordinate (x, y, z).

    Returns:
        A NumPy array of shape [..., 3], where each row represents a spherical
        coordinate (r, theta, phi).
    """

    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    r = np.linalg.norm(cartesian_coordinates, axis=-1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack([r, theta, phi], axis=-1)


def get_ideal_normal_ball(size):

    """
    UNIT-TESTED
    BLENDER CONVENTION
    X: forward
    Y: right
    Z: up

    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(1, -1, size)

    #use indexing 'xy' torch match vision's homework 3
    y,z = torch.meshgrid(y, z ,indexing='xy')

    x = (1 - y**2 - z**2)
    mask = x >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask

    # get real z value
    x = torch.sqrt(x)

    # clean up normal map value outside mask
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask


def get_ideal_normal_ball_np(size):

    """
    UNIT-TESTED
    BLENDER CONVENTION
    X: forward
    Y: right
    Z: up

    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    y = np.linspace(-1, 1, size)
    z = np.linspace(1, -1, size)

    #use indexing 'xy' torch match vision's homework 3
    y, z = np.meshgrid(y, z, indexing='xy')

    x = (1 - y**2 - z**2)
    mask = x >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask

    # get real z value
    x = np.sqrt(x)

    # clean up normal map value outside mask
    normal_map = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

    return normal_map, mask


def get_reflection_vector_map(I: np.array, N: np.array):
    """
    UNIT-TESTED
    Args:
        I (np.array): Incoming light direction #[None,None,3]
        N (np.array): Normal map #[H,W,3]
    @return
        R (np.array): Reflection vector map #[H,W,3]
    """

    # R = I - 2((I⋅ N)N) #https://math.stackexchange.com/a/13263
    dot_product = (I[...,None,:] @ N[...,None])[...,0]
    R = I - 2 * dot_product * N
    return R


def get_SH_coefficients(l, dirs, real=True):
    """
    Compute the spherical harmonic coefficients for a given set of directions.

    Parameters:
    - l (int): Degree of the spherical harmonics (maximum l).
    - dirs (ndarray): Array of directions with shape (num_directions, 2), where each row is [phi, theta] in radians.
    - real (bool): Whether to return only the real part of the spherical harmonics (default: True).

    Returns:
    - coeff (ndarray): Array of shape (num_coefficients, num_directions) containing the SH coefficients.
    """
    num_coefficients = (l + 1) ** 2
    num_directions = dirs.shape[0]

    # Initialize the coefficients array (real part only, avoid using complex numbers)
    coeff = np.zeros((num_coefficients, num_directions), dtype=np.float64)

    idx = 0
    # Loop through all degrees and orders of the spherical harmonics
    for l_idx in range(l + 1):
        for m_idx in range(-l_idx, l_idx + 1):
            # Compute the spherical harmonic for this (l, m) pair
            sh_value = sph_harm(m_idx, l_idx, dirs[:, 0], dirs[:, 1])

            # Only take the real part if required
            coeff[idx, :] = sh_value.real if real else sh_value

            idx += 1

    return coeff


def getSH(em_img, grayscale=True, l=2, basisType=True):
    '''
    em_img: a given environment map [h * w * 3]
    EV: exposure value
    l: degree of sh
    dirs:   [azimuth_1 inclination_1; ...; azimuth_K inclination_K] angles
            in rads for each evaluation point, where inclination is the
            polar angle from zenith: inclination = pi/2-elevation
    basisType:  'complex' or 'real' spherical harmonics

    Returns:
    sh_coeff: [3 * (l+1)**2] SH coefficients of environment map
    sh_img: [h * w * 3] image approximated by SH
    '''
    # get direction of each pixel
    em_img = np.asarray(em_img)

    if max(em_img.shape) > 1024:
        em_img = em_img[:, -512:, :]

    if grayscale:
        em_img = cv2.cvtColor(em_img, cv2.COLOR_RGB2GRAY)
        em_img = np.stack([em_img, em_img, em_img], axis=-1)

    em_img = em_img.astype(np.float64)
    h, w, c = em_img.shape

    #print(em_img.shape) # checked
    theta = (np.arange(h)+0.5)/h*np.pi
    phi = (np.arange(w)+0.5)/w*2*np.pi
    x, y = np.meshgrid(phi, theta)
    dirs = np.vstack((x.ravel(), y.ravel())).T

    # computes SH basis value of different directions
    #coeff_mat = get_sh_debug(l, dirs, basisType)
    sh_num = (l + 1) ** 2
    coeff = get_SH_coefficients(l, dirs)

    # computes differential of solid angle
    theta = np.arange(h + 1) * np.pi / h
    val = np.cos(theta)
    w_theta = val[:-1] - val[1:]
    val = np.arange(w + 1) * 2 * np.pi / w
    w_phi = val[1:] - val[:-1]
    x_s, y_s = np.meshgrid(w_phi, w_theta)
    d_omega = (x_s.ravel() * y_s.ravel())

    # computes SH coeff
    r = em_img[:, :, 0].ravel() * d_omega
    g = em_img[:, :, 1].ravel() * d_omega
    b = em_img[:, :, 2].ravel() * d_omega

    r_coeff = np.sum(np.tile(r[:, np.newaxis], (1, sh_num)).T * coeff, axis=1)
    g_coeff = np.sum(np.tile(g[:, np.newaxis], (1, sh_num)).T * coeff, axis=1)
    b_coeff = np.sum(np.tile(b[:, np.newaxis], (1, sh_num)).T * coeff, axis=1)

    sh_coeff = np.vstack((r_coeff, g_coeff, b_coeff))

    out_r = np.sum(coeff * np.tile(r_coeff, (h * w, 1)).T, axis=0)
    out_g = np.sum(coeff * np.tile(g_coeff, (h * w, 1)).T, axis=0)
    out_b = np.sum(coeff * np.tile(b_coeff, (h * w, 1)).T, axis=0)

    sh_img = np.zeros((h, w, 3), dtype=np.uint8)
    sh_img[:, :, 0] = np.reshape(out_r, (h, w))
    sh_img[:, :, 1] = np.reshape(out_g, (h, w))
    sh_img[:, :, 2] = np.reshape(out_b, (h, w))
    sh_img[np.where(sh_img>=220)] = 0

    return sh_coeff, sh_img


def create_light_mask(gaussian_centers, canva_size, kernel_size=0, radius=30, sigma=10):
    '''
    Returns: a gaussian light mask image with smooth transition
    '''
    canva = np.zeros(canva_size, dtype=np.float32)
    summed_blur = np.zeros_like(canva, dtype=np.float32)
    for cntr in gaussian_centers:
        # apply multiple Gaussian_blurs and sum the results
        light_mask = cv2.circle(canva.copy(), cntr, radius, (1,), thickness=-1)
        light_mask = cv2.GaussianBlur(light_mask, (kernel_size, kernel_size), sigma).astype(np.float32)
        #light_mask = np.clip(light_mask * 255, 0, 255).astype(np.uint8)
        summed_blur += light_mask * 255
    
    # normalize summed light spots
    light_mask = np.clip(summed_blur, 0, 255).astype(np.uint8)
    return light_mask


def create_inverse_gaussian_light_mask(image_size, center, radius=50, sigma=30):
    """
    Create an inverse Gaussian light mask where the edges are bright and the center is dark.
    
    Parameters:
    - image_size: tuple (height, width) for the size of the mask.
    - center: tuple (x, y) representing the center of the light spot.
    - radius: radius of the light area (affects the spread of the mask).
    - sigma: standard deviation for the Gaussian blur.
    
    Returns:
    - An inverse Gaussian light mask image with smooth transition.
    """
    # Create a grid of x, y coordinates
    x = np.linspace(0, image_size[1] - 1, image_size[1])
    y = np.linspace(0, image_size[0] - 1, image_size[0])
    x, y = np.meshgrid(x, y)

    # Calculate the squared distance from the center point
    dist_sq = (x - center[0])**2 + (y - center[1])**2

    # Create a Gaussian distribution based on distance from center
    gaussian_mask = np.exp(-dist_sq / (2 * sigma**2))

    # Invert the Gaussian to make edges bright and center dark
    inverse_gaussian_mask = 1 - gaussian_mask

    # Normalize the mask to the 0-255 range
    inverse_gaussian_mask = np.clip(inverse_gaussian_mask * 255, 0, 255).astype(np.uint8)

    return inverse_gaussian_mask


def angle_from_centroid(points, centroid=(128, 128)):
    ang_deg = []
    for point in points:
        x, y = point
        cx, cy = centroid
        ang_deg.append(atan2(y - cy, x - cx))
    
    ang_ind = sorted(range(len(ang_deg)), key=lambda i: ang_deg[i], reverse=False)
    #print("angle ccw in range[-\pi, \pi]", ang_ind, ang_deg)
    return ang_ind, np.array(ang_deg)


def group_by_angle(pts, strength):
    valid_pts = []
    ang_ind, ang_deg = angle_from_centroid(pts)
    for ang in range(-2, 2):
        ind = np.where((ang_deg>=ang*pi/2) & (ang_deg<=(ang+1)*pi/2))[0]
        if ind.size > 1: 
            pts_quadrant = np.take(pts, ind, axis=0) # percentile 
            strength_quadrant = np.take(strength, ind)
            #print(f"indexing: {pts_quadrant}, {pts}, {ind}, {strength_quadrant}")
            pt = pts_quadrant[np.argmax(strength_quadrant)]
            valid_pts.append(pt)
        elif ind.size == 1:
            #print(f"indexing: {pts[ind[0]]}")
            valid_pts.append(pts[ind[0]])
    
    return valid_pts