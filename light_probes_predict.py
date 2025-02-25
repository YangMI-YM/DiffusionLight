from glob import glob
import os
import shutil
from scipy.special import sph_harm
import cv2
from PIL import Image
import pillow_avif # handling avif format image
import torch
import numpy as np
import skimage
from tqdm import tqdm
from scipy.special import legendre, eval_legendre, lpmn, factorial
import scipy.constants as const
from skimage import data, exposure, img_as_float
from operator import itemgetter
from math import atan2

from utils import *


def hemi_sphere_light_src(sphere_src, val_dist=50, ball_dilate=10, mean_thres=140, thres_dist=6.0):
    '''
    sphere_src: path to chrome ball
    val_dist: for light mask pending
    ball_dilate: pending
    mean_thres: predefined light source threshold, if maxVal less than the thres then increase the ev value
    returns (point_light, directional_light)
    '''
    point_light = []
    directional_light = []
    bound_enh = 2

    ## original image
    org_src =  glob('/content/drive/MyDrive/Light_SH/light_source_test/'+ img_name.split('_ev')[0] + '.*')
    bg_img = Image.open(org_src[0])
    bg_img = pil_square_crop_image(bg_img, (1024, 1024))
    bg_img = np.asarray(bg_img)

    ## light source in chrome ball
    spb = Image.open(sphere_src)
    spb = np.asarray(spb)
    illu = cv2.cvtColor(spb, cv2.COLOR_BGR2GRAY)
    # Gaussian blurring
    illu = cv2.GaussianBlur(illu, (5, 5), 0)
    dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2, (0, 0, 0), -1)
    pts_illu = illu - dir_illu

    # cluster close contours
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pts_illu)

    if maxVal < mean_thres:
        clustered_contours = []
    else:
        while maxVal - minVal < val_dist:
            val_dist = val_dist // 2
        _, illu_mask = cv2.threshold(pts_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
        illu_cnts, _ = cv2.findContours(illu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

        clustered_contours, ring_contours = agglomerative_cluster(list(illu_cnts), threshold_distance=thres_dist) # 8.0
        print("num of cnts", len(illu_cnts), len(clustered_contours))

    while len(clustered_contours) == 0 or maxVal < mean_thres:
        _ev = sphere_src.split('_ev-')[-1].split('.')[0]
        if _ev == '00':
            break
        _ev_enh = '_ev-' + str(int(_ev) - 10) # enhanced ev
        sphere_src = sphere_src.replace('_ev-' + _ev, _ev_enh)
        print(sphere_src)
        if not os.path.exists(sphere_src):
            return None, None
        spb = Image.open(sphere_src)
        spb = np.asarray(spb)
        illu = cv2.cvtColor(spb, cv2.COLOR_BGR2GRAY)
        # Gaussian blurring
        illu = cv2.GaussianBlur(illu, (5, 5), 0)
        dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2-bound_enh, (0, 0, 0), -1)
        pts_illu = illu - dir_illu

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pts_illu) # IMPORTANT: update thres!
        _, illu_mask = cv2.threshold(pts_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
        illu_cnts, _ = cv2.findContours(illu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        clustered_contours, ring_contours = agglomerative_cluster(list(illu_cnts), threshold_distance=thres_dist) # 8.0
        bound_enh += 5

    if len(clustered_contours) == 0:
        bg_ball = envmap2ball(bg_img, mirror=True)
        ev_ball = exposure.adjust_gamma(img_as_float(bg_ball), 5) # TBC
        ev_ball = (ev_ball*255).astype(np.uint8)
        bg_illu = cv2.cvtColor(ev_ball, cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(bg_illu)
        _, light_mask = cv2.threshold(bg_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            cntr = (int(center[0]), int(center[1]))

            cv2.circle(bg_illu, cntr, 5, (0, 255, 0), 3)
            #pixel_light.append(cntr)
            #pts_light_area.append(cnt)

    #display(Image.fromarray(pts_illu))
    tmp = np.zeros(pts_illu.shape, dtype=np.uint8)
    for cnt in clustered_contours:
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
    #display(HTML("<h3>Light mask: </3> " + img_name))
    #display(Image.fromarray(tmp))

    is_symmetric = if_symmetric(tmp)
    print(is_symmetric)

    # secure point light source (sorted by y) # who needs ace
    valid_light_src = 0
    _, max_temperature, _, max_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp))
    if max_loc[1] > 128:
        top_hemi = False
    else:
        top_hemi = True
    print(f"max loc: {max_temperature} {max_loc}")
    # TODO if len(clustered_contours) > 4:
    if len(clustered_contours) > 0:
        #clustered_contours = sorted(clustered_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        clustered_contours = sorted(clustered_contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * pts_illu.shape[1] )  # logic: x+y*w
        for cnt in clustered_contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            tmp_mask = np.zeros(pts_illu.shape, dtype=np.uint8)
            cv2.drawContours(tmp_mask, [cnt], -1, 255, -1)
            _, highlight, _, highlight_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp_mask))

            # mean val within contour area /
            mean_circle_cntr = np.mean(pts_illu[center[1]-5:center[1]+5, center[0]-5:center[0]+5])
            mean_maxval = np.mean(pts_illu[highlight_loc[1]-5:highlight_loc[1]+5, highlight_loc[0]-5:highlight_loc[0]+5])
            print(center, mean_circle_cntr, highlight, highlight_loc, mean_maxval)
            cntr = (int(center[0]), int(center[1])) if mean_circle_cntr > mean_maxval-10 else highlight_loc
            if min(cntr[1], highlight_loc[1]) > max_loc[1] > 128 or max_temperature-highlight > 20:
                continue
            if top_hemi and min(cntr[1], highlight_loc[1]) > 128:
                continue
            valid_light_src += 1
            cv2.circle(pts_illu, cntr, 5, (0, 255, 0), -1)
    #display(HTML("<h3> Point Light Center: </3> " + img_name))
    #display(Image.fromarray(cv2.cvtColor(pts_illu, cv2.COLOR_GRAY2RGB)))

    # secure directional light source
    if len(ring_contours) > 0:
        ring_contours = np.vstack(ring_contours)
        ring_bound = cv2.minAreaRect(ring_contours)
        ring_center = (int(ring_bound[0][0]), int(ring_bound[0][1]))
        top_left = (int(ring_center[0]-ring_bound[1][0]/2), int(ring_center[1]-ring_bound[1][1]/2))
        buttom_right = (int(ring_center[0]+ring_bound[1][0]/2), int(ring_center[1]+ring_bound[1][1]/2))
        #print(top_left, buttom_right)

        # contours cross quanters
        if top_left[0] < 128 and top_left[1] < 128 and buttom_right[0] > 128 and buttom_right[1] > 128:
            cv2.circle(pts_illu, ring_center, 5, (0, 0, 255), 3)
            #display(HTML("<h3>Directional Light Center: </3> " + img_name))
            #display(Image.fromarray(cv2.cvtColor(pts_illu, cv2.COLOR_GRAY2RGB)))

    # save to output path
    output_tile = os.path.dirname(os.path.dirname(sphere_src)).replace('_light', '_dir_est')
    if not os.path.exists(output_tile):
        os.makedirs(output_tile)
    illus = vconcat_resize_min([cv2.cvtColor(pts_illu, cv2.COLOR_GRAY2RGB), cv2.cvtColor(dir_illu, cv2.COLOR_GRAY2RGB)])
    tile = hconcat_resize_min([bg_img, illus])

    cv2.imwrite(os.path.join(output_tile, img_name.split('_ev')[0]+'.png'), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    # ends here. early stop
    return clustered_contours, ring_contours # clustering


    top_illu = pts_illu.copy()
    top_illu[128:, :] = 0
    buttom_illu = pts_illu.copy() # more like reflectant
    buttom_illu[:128, :] = 0
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(top_illu)

    while maxVal < mean_thres:
        _ev = sphere_src.split('_ev-')[-1].split('.')[0]
        if _ev == '00':
            break

        _ev_enh = '_ev-' + str(int(_ev) - 10) # enhanced ev
        sphere_src = sphere_src.replace('_ev-' + _ev, _ev_enh)
        print(sphere_src)
        if not os.path.exists(sphere_src):
            return None, None
        spb = Image.open(sphere_src)
        spb = np.asarray(spb)
        illu = cv2.cvtColor(spb, cv2.COLOR_BGR2GRAY)
        # Gaussian blurring
        illu = cv2.GaussianBlur(illu, (5, 5), 0)
        dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2, (0, 0, 0), -1)
        pts_illu = illu - dir_illu
        top_illu = pts_illu.copy()
        top_illu[128:, :] = 0
        #bottom_illu = pts_illu[:128, :] = 0 # typically not
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(top_illu)

    # point lights top-hemi
    # mind that gamma affect the medium val
    #median = pts_illu[pts_illu>0].mean()
    #print("median value", median, maxVal)
    _, light_mask = cv2.threshold(top_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
    pixel_light = []
    pts_light_area = []
    dir_light = []

    light_mask = cv2.dilate(light_mask, kernel=(3, 3), iterations=5) # force to merge adjacent contours
    display(Image.fromarray(light_mask)) # the best iteration num required
    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(contours[0])
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        tmp = np.zeros(light_mask.shape, dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        highlight_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp))[-1]
        #center, radius = cv2.minEnclosingCircle(cnt)
        if cv2.contourArea(cnt) < 1e-1 * max_area:
            continue
        # light distance to sphere center
        #print(f"light dist to sphere center{np.sqrt((center[0]-128)**2 + (center[1]-128)**2)}")
        if np.sqrt((center[0]-128)**2 + (center[1]-128)**2) > 128 - ball_dilate + 1:
            dir_light.append(highlight_loc)
            continue
        # mean val within contour area /
        mean_circle_cntr = np.mean(pts_illu[center[0]-5:center[1]+5, center[1]-5:center[1]+5])
        mean_maxval = np.mean(pts_illu[maxLoc[0]-5:maxLoc[0]+5, maxLoc[1]-5:maxLoc[1]+5])

        cntr = (int(center[0]), int(center[1])) if mean_circle_cntr > mean_maxval else highlight_loc
        #if center[1] > 128:
        #    continue
        cv2.circle(pts_illu, cntr, 5, (0, 255, 0), 3)
        pixel_light.append(cntr)
        pts_light_area.append(cnt)

    # point light buttom-hemi # what if maxVal-val_dist < minVal
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(buttom_illu)
    if maxVal - val_dist < minVal:
        print(f"illu val underflow!")

    _, light_mask = cv2.threshold(buttom_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
    light_mask = cv2.dilate(light_mask, kernel=(3, 3), iterations=6) # force to merge adjacent contours
    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(contours[0])
    for cnt in contours:
        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        tmp = np.zeros(light_mask.shape, dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        highlight_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp))[-1]

        if cv2.contourArea(cnt) < 1e-1 * max_area:
            continue
        # light distance to sphere center

        if np.sqrt((center[0]-128)**2 + (center[1]-128)**2) > 128 - ball_dilate + 1:
            dir_light.append(highlight_loc)
            continue
        # mean val within contour area /
        mean_circle_cntr = np.mean(pts_illu[center[0]-5:center[1]+5, center[1]-5:center[1]+5])
        mean_maxval = np.mean(pts_illu[maxLoc[0]-5:maxLoc[0]+5, maxLoc[1]-5:maxLoc[1]+5])

        cntr = (int(center[0]), int(center[1])) if mean_circle_cntr > mean_maxval else highlight_loc
        #if center[1] < 128: # mirror!
        #    continue
        cv2.circle(pts_illu, cntr, 5, (0, 0, 255), 3)
        pixel_light.append(cntr)
        pts_light_area.append(cnt)

    if len(pixel_light) == 0 and len(dir_light) < 3:
        mirror = True
        # try finding light source from org img
        print(pixel_light, img_name.split('_ev'))
        org_src =  glob('/content/drive/MyDrive/Light_SH/light_source_test/'+ img_name.split('_ev')[0] + '.*')
        bg_img = Image.open(org_src[0])
        bg_img = pil_square_crop_image(bg_img, (1024, 1024))
        bg_img = np.asarray(bg_img)
        bg_ball = envmap2ball(bg_img, mirror=True)
        ev_ball = exposure.adjust_gamma(img_as_float(bg_ball), 5) # TBC
        ev_ball = (ev_ball*255).astype(np.uint8)
        bg_illu = cv2.cvtColor(ev_ball, cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(bg_illu)
        _, light_mask = cv2.threshold(bg_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #center, radius = cv2.minEnclosingCircle(cnt)
            #if radius < 10 or cv2.contourArea(cnt) < 20:
            #    continue
            cntr = (int(center[0]), int(center[1]))
            if center[1] > 128:
                continue
            cv2.circle(bg_illu, cntr, 5, (0, 255, 0), 3)
            pixel_light.append(cntr)
            pts_light_area.append(cnt)
        #display(Image.fromarray(bg_img))
        #display(HTML("<h3>Original Input: </3> " + img_name))
        #display(bg_ball)
        #display(bg_illu)
    #else:
    #    display(HTML("<h3>Point light: </3> " + img_name))
    #    display(pts_illu)

    #collect point lights where y ~ 128
    matching_pts = filter(lambda x: 128-x[1]<5, pixel_light)
    print(f"collection points near the horizon {list(matching_pts).count(True)}")

    # transform pixel-based light position to world geometry coordinate (x, y) ONLY
    world_light = transform2Dpos2Spherical3D(pixel_light)
    print(f"light src pos in world coord ({int(world_light[0])}, {int(world_light[1])})")

    if len(dir_light) >= 3:
        print(f"directional light exists")

    #display(dir_illu)
    '''
    # if directional light
    _, light_mask = cv2.threshold(dir_illu, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dir_light_area = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        dist_set = [contour_dist(cnt, pt_light_cnt)[0] for pt_light_cnt in pts_light_area]
        print(f"closest distance between point light and directional light {dist_set}")
        if len(np.where(np.array(dist_set) < 10)):
            continue
        # what is the representation of directional light (x, y, z)
        light_dir = np.array([1, 1, 1]) / np.sqrt(3)
    # cart2spherical

    '''
    return point_light, directional_light


def avg_hemi_sphere_light_src(sphere_src, vis_dir, val_dist=50, ball_dilate=10, mean_thres=140, thres_dist=6.0):
    '''
    sphere_src: path to chrome ball
    val_dist: for light mask pending
    ball_dilate: pending
    mean_thres: predefined light source threshold, if maxVal less than the thres then increase the ev value
    returns (point_light, directional_light)
    '''
    point_light = []
    directional_light = []
    bound_enh = 2

    spb = []
    ev_list = []
    ## light source in chrome balls
    ball_list = glob(sphere_src.split('_ev-')[0] + '_ev-*')

    for sphere_ball in ball_list:
        _ev = sphere_ball.split('_ev-')[-1].split('.')[0]
        ev_ball = Image.open(sphere_ball)
        ev_ball = np.asarray(ev_ball)[:, :, :3]
        illu = cv2.cvtColor(ev_ball, cv2.COLOR_BGR2GRAY)
        if exposure.is_low_contrast(illu): # not ideal for feature detection
            continue
        spb.append(illu)
        #ev_list.append(1)

        ev_list.append(2-float(_ev)/50) # [1.9, 0.1, 1]

    illu = equal_blend(spb, ev_list)
    #display(HTML("<h3>Average Light Probes: </3> " + img_name))
    #display(Image.fromarray(illu))

    # Gaussian blurring
    illu = cv2.GaussianBlur(illu, (5, 5), 0)
    dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2, (0, 0, 0), -1)
    pts_illu = illu - dir_illu

    # cluster closed contours
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pts_illu)
    print("max value after blending", maxVal)

    while maxVal - minVal < val_dist:
        val_dist = val_dist // 2
    _, illu_mask = cv2.threshold(pts_illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
    illu_cnts, _ = cv2.findContours(illu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    clustered_contours, ring_contours = agglomerative_cluster(list(illu_cnts), threshold_distance=thres_dist*2) # 8.0
    print("num of cnts", len(illu_cnts), len(clustered_contours))
    if len(clustered_contours) == 0: # 2nd chance to find light spots
        # use ev-50 instead 
        spb5 = Image.open(sphere_src)
        spb5 = np.asarray(spb5)
        #print("***", spb5.shape)
        illu = cv2.cvtColor(spb5, cv2.COLOR_BGR2GRAY)
        # Gaussian blurring
        illu = cv2.GaussianBlur(illu, (5, 5), 0)
        dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2-bound_enh, (0, 0, 0), -1)
        pts_illu = illu - dir_illu
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pts_illu)
        _, light_mask = cv2.threshold(illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
        illu_cnts, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clustered_contours, ring_contours = agglomerative_cluster(list(illu_cnts), threshold_distance=thres_dist)
    if len(clustered_contours) == 0:
        # expand search range of light TBC
        print(f"No light detected {sphere_src}")

    tmp = np.zeros(pts_illu.shape, dtype=np.uint8)
    for cnt in clustered_contours:
        cv2.drawContours(tmp, [cnt], -1, 255, -1)

    # secure point light source (sorted by y) # who needs ace
    valid_light_src = 0
    _, max_temperature, _, max_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp))

    if max_loc[1] > 128:
        top_hemi = False
    else:
        top_hemi = True
    print(f"max temp & loc: {max_temperature} {max_loc}")

    if len(clustered_contours) > 0:
        #clustered_contours = sorted(clustered_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        clustered_contours = sorted(clustered_contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * pts_illu.shape[1] )  # logic: x+y*w
        for cnt in clustered_contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            tmp_mask = np.zeros(pts_illu.shape, dtype=np.uint8)
            cv2.drawContours(tmp_mask, [cnt], -1, 255, -1)
            _, highlight, _, highlight_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp_mask))

            # mean val within contour area / #TODO 
            mean_circle_cntr = np.mean(pts_illu[center[1]-5:center[1]+5, center[0]-5:center[0]+5])
            mean_maxval = np.mean(pts_illu[highlight_loc[1]-5:highlight_loc[1]+5, highlight_loc[0]-5:highlight_loc[0]+5])
            print(center, mean_circle_cntr, highlight, highlight_loc, mean_maxval)
            cntr = (int(center[0]), int(center[1])) if mean_circle_cntr > mean_maxval-10 else highlight_loc
            if min(cntr[1], highlight_loc[1]) > max_loc[1] > 128 or max_temperature-highlight > 20:
                continue
            if top_hemi and min(cntr[1], highlight_loc[1]) > 128:
                continue
            valid_light_src += 1
            cv2.circle(pts_illu, cntr, 5, (0, 255, 0), -1)
            point_light.append(cntr)
        
        #print(os.path.join(vis_dir, os.path.basename(sphere_src).split('_ev-')[0]+'.png'))
        #cv2.imwrite(os.path.join(vis_dir, os.path.basename(sphere_src).split('_ev-')[0]+'.png'), pts_illu)
    #display(HTML("<h3> Point Light Center: </3> " + img_name))
    #display(Image.fromarray(cv2.cvtColor(pts_illu, cv2.COLOR_GRAY2RGB)))

    # secure directional light source
    if len(clustered_contours) == 0 and len(ring_contours) > 0:
        ring_contours = np.vstack(ring_contours)
        ring_bound = cv2.minAreaRect(ring_contours)
        ring_center = (int(ring_bound[0][0]), int(ring_bound[0][1]))
        top_left = (int(ring_center[0]-ring_bound[1][0]/2), int(ring_center[1]-ring_bound[1][1]/2))
        buttom_right = (int(ring_center[0]+ring_bound[1][0]/2), int(ring_center[1]+ring_bound[1][1]/2))
        #print(top_left, buttom_right)

        # contours cross quanters
        if top_left[0] < 128 and top_left[1] < 128 and buttom_right[0] > 128 and buttom_right[1] > 128:
            cv2.circle(pts_illu, ring_center, 5, (0, 0, 255), 3)
            
        directional_light.append(ring_center)

    # transform pixel-based light position to world geometry coordinate (x, y) ONLY
    if len(point_light) > 0:
        world_light = transform2Dpos2Spherical3D(point_light)
        print(f"light src pos in world coord ({int(world_light[0][0])}, {int(world_light[0][1])})")
        # world light mask
        light_mask = create_light_mask(point_light, (256, 256))
        # tile spb & light mask
        tile = hconcat_resize_min([pts_illu, light_mask])
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(sphere_src).split('_ev-')[0]+'_tile.png'), tile)
    else: # TODO remove samples without light spots
        light_mask = create_inverse_gaussian_light_mask(center=(128, 128), image_size=(256, 256)) # dark center 
        tile = hconcat_resize_min([pts_illu, light_mask])
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(sphere_src).split('_ev-')[0]+'_tile.png'), tile)

    return clustered_contours, ring_contours


def avg_sphere_bright(sphere_src, vis_dir, val_dist=50, ball_dilate=10, mean_thres=140, thres_dist=6.0):
    '''
    sphere_src: path to chrome ball
    val_dist: for light mask pending
    ball_dilate: pending
    mean_thres: predefined light source threshold, if maxVal less than the thres then increase the ev value
    returns (point_light, directional_light)
    '''
    point_light = []
    directional_light = []
    bound_enh = 2
    min_area = 1e2
    threshold_factor = 1.5

    spb = []
    ev_list = []
    ## light source in chrome balls
    ball_list = glob(sphere_src.split('_ev-')[0] + '_ev-*')

    for sphere_ball in ball_list:
        _ev = sphere_ball.split('_ev-')[-1].split('.')[0]
        ev_ball = Image.open(sphere_ball)
        ev_ball = np.asarray(ev_ball)[:, :, :3]
        illu = cv2.cvtColor(ev_ball, cv2.COLOR_BGR2GRAY)
        if exposure.is_low_contrast(illu): # not ideal for feature detection
            continue
        spb.append(illu)
        #ev_list.append(1)

        ev_list.append(2-float(_ev)/50) # [1.9, 0.1, 1]

    illu = equal_blend(spb, ev_list)
    

    # Step 1: Gaussian blurring
    illu = cv2.GaussianBlur(illu, (5, 5), 0)
    dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2, (0, 0, 0), -1)
    pts_illu = illu - dir_illu

    # Step 2: local thresholding based on smoothed sphere ball
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pts_illu)
    mean_val = np.mean(pts_illu)
    std_val = np.std(pts_illu)
    thres_val = mean_val + threshold_factor * std_val

    # Step 3:
    _, thres_illu = cv2.threshold(pts_illu, thres_val, 255, cv2.THRESH_BINARY)

    # Step 4: Find connected components (bright regions)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thres_illu, 8, cv2.CV_32S)
    
    bright_areas = []
    # Step 5: Filter out small noisy components by area
    for i in range(1, num_labels):  # Start from 1 to ignore background
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            # Extract bounding box of the bright area
            x, y, w, h, _ = stats[i]
            bright_areas.append((x, y, w, h))
    
    clustered_contours = bright_areas
    if len(clustered_contours) == 0: # 2nd chance to find light spots
        # use ev-50 instead 
        spb5 = Image.open(sphere_src)
        spb5 = np.asarray(spb5)
        #print("***", spb5.shape)
        illu = cv2.cvtColor(spb5, cv2.COLOR_BGR2GRAY)
        # Gaussian blurring
        illu = cv2.GaussianBlur(illu, (5, 5), 0)
        dir_illu = cv2.circle(illu.copy(), (128, 128), 128-ball_dilate//2-bound_enh, (0, 0, 0), -1)
        pts_illu = illu - dir_illu
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pts_illu)
        _, light_mask = cv2.threshold(illu, maxVal-val_dist, 255, cv2.THRESH_BINARY)
        illu_cnts, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clustered_contours, ring_contours = agglomerative_cluster(list(illu_cnts), threshold_distance=thres_dist)
    if len(clustered_contours) == 0:
        # expand search range of light TBC
        print(f"No light detected {sphere_src}")

    print("bright area", clustered_contours)
    # secure point light source (sorted by y) # who needs ace
    tmp = np.zeros(pts_illu.shape, dtype=np.uint8)
    for cnt in clustered_contours:
        #cv2.drawContours(tmp, [cnt], -1, 255, -1)
        cv2.rectangle(tmp, (cnt[0], cnt[1]), (cnt[0]+cnt[2], cnt[1]+cnt[3]), 255, -1)
    valid_light_src = 0
    _, max_temperature, _, max_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp))

    if max_loc[1] > 128:
        top_hemi = False
    else:
        top_hemi = True
    print(f"max temp & loc: {max_temperature} {max_loc}")
    
    if len(clustered_contours) > 0:
        #clustered_contours = sorted(clustered_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
        clustered_contours = sorted(clustered_contours, key=lambda ctr: ctr[0] + ctr[1] * ctr[2] )  # logic: x+y*w
        for cnt in clustered_contours:
            center = (cnt[0]+cnt[2]//2, cnt[1]+cnt[3]//2)
            tmp_mask = np.zeros(pts_illu.shape, dtype=np.uint8)
            cv2.rectangle(tmp_mask, (cnt[0], cnt[1]), (cnt[0]+cnt[2], cnt[1]+cnt[3]), 255, -1)
            _, highlight, _, highlight_loc = cv2.minMaxLoc(cv2.bitwise_and(pts_illu, pts_illu, mask=tmp_mask))

            # mean val within contour area / #TODO 
            mean_circle_cntr = np.mean(pts_illu[center[1]-5:center[1]+5, center[0]-5:center[0]+5])
            mean_maxval = np.mean(pts_illu[highlight_loc[1]-5:highlight_loc[1]+5, highlight_loc[0]-5:highlight_loc[0]+5])
            print(center, mean_circle_cntr, highlight, highlight_loc, mean_maxval)
            cntr = (int(center[0]), int(center[1])) if mean_circle_cntr > mean_maxval-10 else highlight_loc
            if min(cntr[1], highlight_loc[1]) > max_loc[1] > 128 or max_temperature-highlight > 20:
                continue
            if top_hemi and min(cntr[1], highlight_loc[1]) > 128:
                continue
            valid_light_src += 1
            cv2.circle(pts_illu, cntr, 5, (0, 255, 0), -1)
            point_light.append(cntr)

    # secure directional light source
    if len(clustered_contours) == 0 and len(ring_contours) > 0:
        ring_contours = np.vstack(ring_contours)
        ring_bound = cv2.minAreaRect(ring_contours)
        ring_center = (int(ring_bound[0][0]), int(ring_bound[0][1]))
        top_left = (int(ring_center[0]-ring_bound[1][0]/2), int(ring_center[1]-ring_bound[1][1]/2))
        buttom_right = (int(ring_center[0]+ring_bound[1][0]/2), int(ring_center[1]+ring_bound[1][1]/2))
        #print(top_left, buttom_right)

        # contours cross quanters
        if top_left[0] < 128 and top_left[1] < 128 and buttom_right[0] > 128 and buttom_right[1] > 128:
            cv2.circle(pts_illu, ring_center, 5, (0, 0, 255), 3)
            
        directional_light.append(ring_center)

    # transform pixel-based light position to world geometry coordinate (x, y) ONLY
    if len(point_light) > 0:
        world_light = transform2Dpos2Spherical3D(point_light)
        print(f"light src pos in world coord ({int(world_light[0][0])}, {int(world_light[0][1])})")
        # world light mask
        light_mask = create_light_mask(point_light, (256, 256))
        # tile spb & light mask
        tile = hconcat_resize_min([pts_illu, light_mask])
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(sphere_src).split('_ev-')[0]+'_tile.png'), tile)
    else: # TODO remove samples without light spots
        light_mask = create_inverse_gaussian_light_mask(center=(128, 128), image_size=(256, 256)) # dark center 
        tile = hconcat_resize_min([pts_illu, light_mask])
        cv2.imwrite(os.path.join(vis_dir, os.path.basename(sphere_src).split('_ev-')[0]+'_tile.png'), tile)

    #return clustered_contours, ring_contours


if __name__ == "__main__":

    search_dir = '/home/yangmi/s3data/light_probs/ball_minitestset' # keep depth shallow
    vis_dir = '/home/yangmi/s3data/light_probs/minitestset_output'
    os.makedirs(vis_dir, exist_ok=True)
    symlink = './output'
    image_filename_list = glob(search_dir+'/*-50.*')#[:100]
    images_path = [os.path.join(search_dir, file_path) for file_path in image_filename_list]
    ball_dilate = 10 # used in inpaint step to make a sharper ball edge
    val_dist = 45 # thresholding light mask
    print(len(images_path))

    # If it's a directory, don't overwrite
    if os.path.isdir(symlink):
        print(f"'{symlink}' is a directory, skipping symlink creation.")
    else:
        # Remove the existing symlink/file if it's not a directory
        if os.path.exists(symlink):
            os.remove(symlink)  # or os.unlink(link_dir) for symlinks
        os.symlink(vis_dir, symlink)
        print(f"Symbolic link created from '{vis_dir}' to '{symlink}'.")

    for item in tqdm(images_path):

        img_name = item.split('/')[-1]
        #if 'da8452138170951.6217e424c2cef' not in img_name: 
        #if 'daa773171583341.6470d2059080d' not in img_name:
        #    continue
        ## DC component from SH coeff
        print(img_name)
        try:
            hdr_evm = Image.open(item.replace('-50', '-00'))
        except:
            try:
                hdr_evm = Image.open(item.replace('-50', '-10'))
            except:
                print(f"IO failure on sample {img_name}.")
                continue
        hdr_evm = np.asarray(hdr_evm)
        sh_coeff, sh_hdr = getSH(hdr_evm, l=3) #l_max=3
        print(f"SH coeff light intensity: {sh_coeff[0][0]}, sh_hdr cutoff {np.max(sh_hdr)} ")

        #pt_light, dir_light = avg_hemi_sphere_light_src(item, vis_dir, val_dist, ball_dilate, mean_thres=150)
        avg_sphere_bright(item, vis_dir, val_dist, ball_dilate, mean_thres=150)