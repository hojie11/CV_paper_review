from os.path import join as ospj

import cv2
import json
import imageio
import numpy as np


def load_blender(root, bkg_white=False, downsample=4, file_skip=4):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(ospj(root, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses =[]
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or file_skip == 0:
            skip = 1
        else:
            skip = file_skip
        
        for frame in meta['frames'][::skip]:
            fname = ospj(root, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(counts)-1)]

    imgs = np.concatenate(all_imgs, 0)
    gt_extrinsic = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if downsample:
        H = int(H//downsample)
        W = int(W//downsample)
        focal = focal/downsample

        imgs_reduced = np.zeros((imgs.shape[0], H, W, imgs.shape[3]))
        for i, img in enumerate(imgs):
            imgs_reduced[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_reduced

    H, W = int(H), int(W)
    gt_intrinsic = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    if bkg_white:
        imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
    else:
        imgs = imgs[..., :3]*imgs[..., -1:]
        

    return imgs, [gt_intrinsic, gt_extrinsic], [H, W, focal], i_split


if __name__ == '__main__':
    imgs, gt_camera_param, hw, i_split = load_blender('../data/nerf_synthetic/lego', downsample=2)
    print()
