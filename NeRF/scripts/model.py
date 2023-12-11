import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation='relu') -> None:
        super().__init__()

        main = [nn.Linear(in_feat, out_feat)]
        main += [nn.ReLU() if activation == 'relu' else nn.Sigmoid()]

        self.main = nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)


class NeRF(nn.Module):
    def __init__(self, x_dim, d_dim, w_dim, num_layers, skips=[4], use_dirs=False) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.d_dim = d_dim
        self.num_layers = num_layers
        self.skips = skips
        self.use_dirs = use_dirs

        self.feature_list = [x_dim] + [w_dim if i not in skips else w_dim + x_dim for i in range(1, num_layers + 1)]

        # layer 0~7 and 5'th layer has skip connection
        for idx in range(num_layers):
            in_feat = self.feature_list[idx]
            out_feat = self.feature_list[idx + 1]
            if idx + 1 in skips:
                out_feat = w_dim
            layer = FullyConnectedLayer(in_feat, out_feat)
            setattr(self, f'fc_{idx}', layer)

        if use_dirs:
            # layer 8
            self.fc_sigma = nn.Linear(w_dim, 1)
            self.fc_feat = nn.Linear(w_dim, w_dim)
            # layer 9
            self.fc_dirs = FullyConnectedLayer(w_dim + d_dim, w_dim//2)
            # layer 10
            self.fc_rgb = FullyConnectedLayer(w_dim//2, 3, activation='sigmoid')
        else:
            self.fc_out = nn.Linear(w_dim, 4)

    def forward(self, xyz, dirs):
        out = xyz
        # layer 0~7
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc_{idx}')
            out = layer(out)
            if idx + 1 in self.skips:
                out = torch.concat([out, xyz], dim=-1)

        if self.use_dirs:
            # layer 8
            sigma = self.fc_sigma(out) # return sigma
            feature = self.fc_feat(out)
            # layer 9
            out = torch.concat([feature, dirs], dim=-1)
            out = self.fc_dirs(out)
            # layer 10
            rgb = self.fc_rgb(out)
        else:
            out = self.fc_out(out)
            rgb, sigma = out[..., :3], out[..., -1]

        return {'rgb' : rgb, 'sigma' : sigma}

    def extra_repr(self):
        return f"rays_xyz's dimension : {self.x_dim} rays_direction's dimension : {self.d_dim}"


def make_model(x_dim, d_dim, w_dim=256, num_layers=8, use_dirs=False, device='cpu'):
    nerf_coarse = NeRF(x_dim, d_dim, w_dim=w_dim, num_layers=num_layers, use_dirs=use_dirs)
    nerf_fine = NeRF(x_dim, d_dim, w_dim=w_dim, num_layers=num_layers, use_dirs=use_dirs)

    return {'coarse' : nerf_coarse.to(device), 'fine' : nerf_fine.to(device)}


if __name__ == '__main__':
    model = NeRF(x_dim=63, d_dim=27)
    xyz = torch.randn((1, 64, 63))
    dirs = torch.randn((1, 64, 27))
    print(model(xyz, dirs))