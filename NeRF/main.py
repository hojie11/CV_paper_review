import argparse
import torch

from scripts.model import make_model
from scripts.positional_encoding import get_positional_encoder
from scripts.load_blender import load_blender


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device} for training')

    images, camera_params, hwf, i_split = load_blender(args.file_path,
                                                       bkg_white=True,
                                                       downsample=args.downsample,
                                                       file_skip=args.file_skip)

    i_train, i_val, i_test = i_split
    img_h, img_w, focal = hwf
    gt_intr, gt_extr = camera_params

    embed_fn, input_ch = get_positional_encoder(args.embed_xyz)
    embed_fn_d, input_ch_d = get_positional_encoder(args.embed_dir)

    models = make_model(x_dim=input_ch, d_dim=input_ch_d, use_dirs=True, device=device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=models['coarse'].parameters(),
                                 betas=[.9, .999])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--embed_xyz', type=int, default=10,
                        help='embedding factor of positional encoding')
    parser.add_argument('--embed_dir', type=int, default=4,
                        help='embedding factor of positional encoding')
    parser.add_argument('--use_dirs', type=bool, default=False,
                        help='set view dependence')

    # training arguments
    parser.add_argument('--downsample', type=int, default=4,
                        help='factor that resize all img to downsample ratio')
    parser.add_argument('--file_skip', type=int, default=4,
                        help='how many imgs dataloader skip to implement dataset')

    # directory for training
    parser.add_argument('--file_path', type=str, default='../data/nerf_synthetic/lego')

    args = parser.parse_args()

    main(args)