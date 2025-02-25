import numpy as np
import torch

from utils import (cartesian_to_spherical, 
                    get_ideal_normal_ball, 
                    get_reflection_vector_map
)


def envmap2ball(env_map, ball_size=256, scale=4, file_name=None, mirror=False):
    normal_ball, _ = get_ideal_normal_ball(ball_size * scale)
    _, mask = get_ideal_normal_ball(ball_size)

    # verify that x of normal is in range [0,1]
    assert normal_ball[:,:,0].min() >= 0
    assert normal_ball[:,:,0].max() <= 1

    # camera is pointing to the ball, assume that camera is othographic as it placing far-away from the ball
    if mirror:
        I = np.array([-1, 0, 0]) # order: [x, y, z]
    else:
        I = np.array([1, 0, 0]) # order: [x, y, z]
    ball_image = np.zeros_like(normal_ball)

    # read environment map as float
    env_map = skimage.img_as_float(env_map)[...,:3]


    reflected_rays = get_reflection_vector_map(I[None,None], normal_ball)
    spherical_coords = cartesian_to_spherical(reflected_rays)

    theta_phi = spherical_coords[...,1:]

    # scale to [0, 1]
    # theta is in range [-pi, pi],
    theta_phi[...,0] = (theta_phi[...,0] + np.pi) / (np.pi * 2)
    # phi is in range [0,pi]
    theta_phi[...,1] = theta_phi[...,1] / np.pi

    if not mirror:
        # mirror environment map because it from inside to outside
        theta_phi = 1.0 - theta_phi

    with torch.no_grad():
        # convert to torch to use grid_sample
        theta_phi = torch.from_numpy(theta_phi[None])
        env_map = torch.from_numpy(env_map[None]).permute(0,3,1,2)
        # grid sample use [-1,1] range
        grid = (theta_phi * 2.0) - 1.0
        ball_image = torch.nn.functional.grid_sample(env_map.float(), grid.float(), mode='bilinear', padding_mode='border', align_corners=True)
        ball_image = ball_image[0].permute(1,2,0).numpy()
        ball_image = np.clip(ball_image, 0, 1)
        ball_image = skimage.transform.resize(ball_image, (ball_image.shape[0] // scale, ball_image.shape[1] // scale), anti_aliasing=True)
        ball_image[~mask] = np.array([0,0,0])
        if file_name is None:
            return (ball_image * 255).astype(np.uint8)
        elif file_name.endswith(".exr"):
            ezexr.imwrite(os.path.join(args.ball_dir, file_name), ball_image.astype(np.float32))
            return None
        else:
            ball_image = skimage.img_as_ubyte(ball_image)
            skimage.io.imsave(os.path.join(args.ball_dir, file_name), ball_image)
            return None