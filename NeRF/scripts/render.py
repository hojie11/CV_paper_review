import torch


def batchify_rays(args, rays, models, embed_fns):
    """
    Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays.shape[0], args.chunk):
        ret = render_rays(args, rays[i:i+args.chunk], models, embed_fns)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.concat(all_ret[k], dim=0) for k in all_ret}
    return all_ret


def render_rays(args, rays_chunk, models, embed_fns):
    device = rays_chunk.device
    embed_fn, embed_fn_d = embed_fns
    # batach size of rays
    N_rays = rays_chunk.shape[0]

    rays_o, rays_d = rays_chunk[:, 0:3], rays_chunk[:, 3:6]
    bounds = rays_chunk[:, 6:8].view(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1] 

    t_vals = torch.linspace(0., 1., steps=args.N_samples, device=device)
    z_vals = near * (1. -t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, args.N_samples])
    # stratified sampling
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., :1], mids], dim=-1)
    t_rand = torch.rand([N_rays, args.N_samples], device=device)
    z_vals = lower + (upper-lower) * t_rand # distance how far the depth is from cam coord of the origin

    # translation(cam to world coord)
    input_xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1) # N_rays, N_samples, 3
    input_xyz_flat = input_xyz.view(-1, 3)
    embed_input_xyz = embed_fn(input_xyz_flat)
    # rotation(cam to world coord)
    input_dir = rays_d.unsqueeze(1).expand(input_xyz.size())
    input_dir_flat = input_dir.contiguous().view(-1, 3)
    embed_input_dir = embed_fn_d(input_dir_flat)
    
    raw = models['coarse'](embed_input_xyz, embed_input_dir)
    rgb_map, depth_map, disp_map, acc_map, weights = raw2out(raw, z_vals, rays_d)

    if args.N_importance > 0:
        rgb_map_c, disp_map_c, acc_map_c = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            args, z_vals_mid, weights[..., 1:-1], args.N_importance, det=(args.perterb == 0.))
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

        # translation(cam to world coord)
        input_xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1) # N_rays, N_samples, 3
        input_xyz_flat = input_xyz.view(-1, 3)
        embed_input_xyz = embed_fn(input_xyz_flat)
        # rotation(cam to world coord)
        input_dir = rays_d.unsqueeze(1).expand(input_xyz.size())
        input_dir_flat = input_dir.contiguous().view(-1, 3)
        embed_input_dir = embed_fn_d(input_dir_flat)

        raw = models['fine'](embed_input_xyz, embed_input_dir)
        rgb_map, depth_map, disp_map, acc_map, weights = raw2out(raw, z_vals, rays_d)

        return {'rgb_coarse_map' : rgb_map_c, 'disp_coarse_map' : disp_map_c,
                'rgb_fine_map' : rgb_map, 'disp_fine_map' : disp_map}
    return {'rgb_map'  : rgb_map, 'disp_map' : disp_map}


def raw2out(raw, z_vals, rays_d):
    def raw2alpha(raw, dists, act_fn=torch.nn.functional.relu):
        return 1. - torch.exp(-act_fn(raw) * dists)
    device = rays_d.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    infinity = torch.Tensor([1e10]).to(device)
    dists = torch.cat([dists, infinity.expand(dists[..., :1].shape)], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    dists = dists.reshape(-1, 1)

    rgb = torch.sigmoid(raw['rgb'])

    alpha = raw2alpha(raw['sigma'], dists)
    # density x transmittance
    transmittance = torch.cumprod(torch.cat(
        [torch.ones((alpha.shape[0], 1)).to(device), 1. -alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * transmittance

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    disp_map = 1. / torch.maximum(1e-10 * torch.ones_like(depth_map),
                                  depth_map / torch.sum(weights, dim=-1))
    disp_map = torch.where(torch.isnan(disp_map),
                           torch.zeros_like(depth_map), disp_map)
    
    scale_factor = 5.
    disp_map = torch.where(disp_map > scale_factor,
                           scale_factor * torch.ones_like(disp_map), disp_map)
    
    acc_map = torch.sum(weights, dim=-1)
    rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))

    return rgb_map, depth_map, disp_map, acc_map, weights


def sample_pdf(args, bins, weights, N_samples, det=False):
    device = weights.device

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
    return samples