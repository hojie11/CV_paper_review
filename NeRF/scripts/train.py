import numpy as np
import torch

from scripts.rays import get_rays


def train(args, i, i_train, imgs, gt_camera_params, hwf, models, embed_fns, loss, optimizer, device):
    intr_param, extr_param = gt_camera_params
    intr_param = torch.from_numpy(intr_param).to(device)
    extr_param = torch.from_numpy(extr_param).to(device)

    i_img = np.random.choice(i_train)
    target_img = torch.from_numpy(imgs[i_img]).type(torch.float32).to(device)
    target_pose = extr_param[i_img, :3, :4]

    rays_o, rays_d = get_rays(hwf, target_pose)

    """
    would be modified
    """

    return None