from os.path import join as ospj
import argparse

import torch

from scripts.load_blender import load_blender
from scripts.model import make_model
from scripts.positional_encoding import get_positional_encoder
from scripts.train import train


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device} for training')

    imgs, camera_params, hwf, i_split = load_blender(args.file_path,
                                                       bkg_white=True,
                                                       downsample=args.downsample,
                                                       file_skip=args.file_skip)

    i_train, i_val, i_test = i_split

    embed_fn, input_ch = get_positional_encoder(args.embed_xyz, device)
    embed_fn_d, input_ch_d = get_positional_encoder(args.embed_dir, device)

    models = make_model(x_dim=input_ch, d_dim=input_ch_d, use_dirs=True, device=device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=models['coarse'].parameters(),
                                 lr=args.lr,
                                 betas=[.9, .999])
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=0.1,
                                                       last_epoch=-1)

    if args.resume_iter > 0 :
        checkpoint = torch.load(ospj(args.checkpoint_dir, f'checkpoint_{args.resume_iter}.ckpt'))
        models['coarse'].load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'RESUME TRAINING : loaded checkpoint from iteration {args.resume_iter}')
    else:
        print('TRAINING FROM SCRATCH')

    for i in range(args.resume_iter, args.total_iters):
        # training
        train(args, i, i_train, imgs, gt_camera_params=camera_params,
              hwf=hwf, models=models, embed_fns=[embed_fn, embed_fn_d], loss=loss, optimizer=optimizer,
              device=device)


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
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate for training')
    parser.add_argument('--total_iters', type=int, default=1000000,
                        help='number of total interations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='iterations to resume training or testing')
    parser.add_argument('--downsample', type=int, default=4,
                        help='factor that resize all img to downsample ratio')
    parser.add_argument('--file_skip', type=int, default=4,
                        help='how many imgs dataloader skip to implement dataset')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')

    # directory for training
    parser.add_argument('--file_path', type=str, default='../data/nerf_synthetic/lego',
                        help='directory for saving input source images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='directory for saving network checkpoints')
    parser.add_argument('--sample_dir', type=str, default='expr/sample',
                        help='directory for saving network checkpoints')
    
    # derectory for testing
    # parser.add_argument('--result_dir', type=str, default='expr/checkpoints',
    #                     help='directory for saving generated images')
    

    args = parser.parse_args()
    main(args)