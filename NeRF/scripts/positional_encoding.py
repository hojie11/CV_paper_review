import torch
import torch.nn as nn


# This code is adapted from "https://github.com/bmild/nerf/blob/master" and modified to pytorch template
class Positional_Encoding(nn.Module):
    def __init__(self, L) -> None:
        super().__init__()
        d = 3
        embed_fns = []
        # include input
        embed_fns.append(lambda x: x)
        out_dim = d


        max_freq = L - 1
        N_freqs = L

        freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.concat([fn(inputs) for fn in self.embed_fns], dim=-1)
    

def get_positional_encoder(L, device):
    embedder_obj = Positional_Encoding(L)
    def embed(x, eo=embedder_obj, device=device): return eo.embed(x).to(device)
    return embed, embedder_obj.out_dim


if __name__ == '__main__':
    pe, pe_dim = get_positional_encoder(10)
    xyz = torch.randn((1, 256, 3))
    pe_xyz = pe(xyz)
    print(pe_xyz, pe_xyz.shape)

    pe, pe_dim = get_positional_encoder(4)
    dir = torch.randn((1, 256, 3))
    pe_dir = pe(dir)
    print(pe_dir, pe_dir.shape)