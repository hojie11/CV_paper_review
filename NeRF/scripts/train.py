import numpy as np
import torch

from scripts.rays import get_rays
from scripts.render import batchify_rays


def train(args, i, i_train, imgs, gt_camera_params, hwf, models, embed_fns, loss, optimizer, device):
    intr_param, extr_param = gt_camera_params
    intr_param = torch.from_numpy(intr_param).to(device)
    extr_param = torch.from_numpy(extr_param).to(device)

    i_img = np.random.choice(i_train)
    target_img = torch.from_numpy(imgs[i_img]).type(torch.float32).to(device)
    target_pose = extr_param[i_img, :3, :4]

    rays_o, rays_d = get_rays(hwf, target_pose)
    if args.use_dirs:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # create ray batch
    rays_o = rays_o.reshape((-1, rays_o.shape[-1]))
    rays_d = rays_d.reshape((-1, rays_d.shape[-1]))
    near = args.near * torch.ones_like(rays_d[..., :1]).to(device)
    far = args.far * torch.ones_like(rays_d[..., :1]).to(device)

    # concat [rays_o, rays_d, near, far] for each ray
    rays = torch.concat((rays_o, rays_d, near, far), dim=-1)

    all_ret = batchify_rays(args, rays, models, embed_fns)
    for k in all_ret:
        k_sh = list(rays_d.shape[:-1] + list(all_ret[k].shape[1:]))
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    rgb, disp, acc = ret_list

    optimizer.zero_grad()
    
    rgb_loss = loss()

    return None