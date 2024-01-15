import numpy as np
import torch


def get_rays(hwf, c2w):
    """Get ray origins, directions from a pinhole camera."""
    device = c2w.get_device()
    H, W, focal = hwf
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                          torch.linspace(0, H - 1, H, device=device))
    i = i.t()
    j = j.t()
    dirs = torch.stack([ (i - W * .5) / focal, 
                        -(j - H * .5) / focal,
                        -torch.ones_like(i)], dim=-1)
    rays_d = dirs @ c2w[:3, :3].T
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


def get_rays_np(hwf, c2w):
    """Get ray origins, directions from a pinhole camera."""
    H, W, focal = hwf
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([ (i - W * .5) / focal, 
                     -(j - H * .5) / focal,
                     -np.ones_like(i)], dim=-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def sample_rays_pixels(rays_o, rays_d, target_img, args):
    img_h, img_w = target_img.size()[:2]

    coords = torch.stack(torch.meshgrid(
                torch.linspace(0, img_h - 1, img_h),
                torch.linspace(0, img_w - 1, img_w)), -1)
    coords = torch.reshape(coords, [-1, 2])  # [ HxW , 2 ]

    # 640000 개 중 1024개 뽑기
    selected_idx = np.random.choice(a=coords.size(
        0), size=args.chunk, replace=False)  # default 1024
    selected_coords = coords[selected_idx].long()  # (N_rand, 2)

    # == Sample Rays ==
    rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
    rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
    # == Sample Pixel ==
    target_img = target_img[selected_coords[:, 0], selected_coords[:, 1]]

    return rays_o, rays_d, target_img  # [1024, 3]
