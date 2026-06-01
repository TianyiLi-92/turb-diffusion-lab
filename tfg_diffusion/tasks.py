import torch as th
from palette_diffusion.palette_diffusion import sum_flat
from guided_diffusion.nn import mean_flat


def load_guider(task):
    """
    Returns log p_0(c|x_0) based on the specified task.
    """
    # For inpainting, p_0(c|x_0) = exp(-||cond - mask * x_0||^2), where cond is 
    # the measurement, and mask = 1 in the measurement region and 0 elsewhere.
    if task == "inpainting":
        def get_guidance(x0, guider_kwargs):
            cond = guider_kwargs["cond"]  # [B, C, H, ...]
            mask = guider_kwargs["mask"]  # [B, C, H, ...]
            return -sum_flat(mask * (cond - x0) ** 2)  / sum_flat(mask)  # [B]

    # For super-resolution with coarse filtering via Haar wavelet, 
    # p_0(c|x_0) = exp(-||ref_approx - x0_approx||^2),
    # where ref_approx and x0_approx are the Haar-approximated (coarse-scale) versions 
    # of the reference and generated signals, respectively.
    elif task == "haar_sr":
        def get_guidance(x0, guider_kwargs):
            level = guider_kwargs["level"]
            reference = guider_kwargs["reference"]  # full-resolution signal [B, C, L]
            with th.no_grad():
                ref_approx = compute_haar_approx(reference, level)  # [B, C, L // 2^level]
            x0_approx = compute_haar_approx(x0, level)  # [B, C, L // 2^level]
            return -sum_flat((ref_approx - x0_approx) ** 2)  # [B]

    # # For threshold_acc, p_0(c|x_0) = exp(-ReLU(threshold - max(acc(t_target)))), 
    # # where max is taken over the acceleration components at t_target, ensuring 
    # # that at least one component exceeds the threshold.
    # elif task == "threshold_acc":
    #     def get_guidance(x0, guider_kwargs):
    #         # Target time index for evaluating acceleration
    #         t_idx = guider_kwargs["t_target"]
    #         assert 0 <= t_idx < x0.shape[-1] - 1, "t_target is out of bounds."
    #         # Acceleration scale factor (used to normalize the threshold)
    #         std_acc = th.tensor(0.0040588385452668034, dtype=th.float32, device=x0.device)
    #         threshold = guider_kwargs["threshold"] * std_acc
    #         # Compute finite-difference acceleration at time t_idx
    #         acc = th.abs(x0[..., t_idx+1] - x0[..., t_idx])  # [B, C]
    #         # Log-Sum-Exp approximation of max(acc)
    #         beta = 10
    #         lse_max = (1/beta) * th.logsumexp(beta*acc, dim=1)  # [B]
    #         # Penalize if the max acceleration is below the threshold
    #         return -th.relu(threshold - lse_max)  # [B]

    # For moment_du_tau, p_0(c|x_0) = exp(<(du_tau)^order>), which promotes a 
    # specific even-order moment of velocity increments at scale tau along a 
    # trajectory, thereby preferentially sampling extreme events.
    elif task == "moment_du_tau":
        def get_guidance(x0, guider_kwargs):
            order, tau = guider_kwargs["order"], guider_kwargs["tau"]
            assert order % 2 == 0, "moment order must be an even number."
            du = x0[..., tau:] - x0[..., :-tau]  # [B, C, H-tau]
            return mean_flat(du ** order)  # [B]

    # For local_moment_du_tau, p0(c|x0) = exp(<(du_tau)^order * g(t)>), which 
    # promotes a specific even-order moment of velocity increments at scale tau, 
    # localized by a filter of the given kind centered at tc with half-width sigma, 
    # along a trajectory, thereby preferentially sampling extreme events.
    elif task == "local_moment_du_tau":
        def get_guidance(x0, guider_kwargs):
            order, tau = guider_kwargs["order"], guider_kwargs["tau"]
            assert order % 2 == 0, "moment order must be an even number."
            kind = guider_kwargs["kind"]
            tc, sigma = guider_kwargs["tc"], guider_kwargs["sigma"]

            du = x0[..., tau:] - x0[..., :-tau]  # [B, C, H-tau]
            t = th.arange(du.shape[2], dtype=du.dtype, device=du.device)

            if kind == "box":
                weights = ((t >= tc - sigma) & (t <= tc + sigma)).to(dtype=du.dtype)
            elif kind == "gaussian":
                weights = th.exp(-0.5 * ((t - tc) / sigma) ** 2)
            else:
                raise ValueError("`kind` must be 'box' or 'gaussian'")
            return mean_flat((du ** order) * weights.view(1, 1, -1))  # [B]

    else:
        raise NotImplementedError
    return get_guidance


def compute_haar_approx(x, level):
    """
    Compute Haar approximation coefficients at given level.
    :param x: a Tensor of shape [B, C, L]
    :param level: int, Haar level
    :return: a Tensor of shape [B, C, L // 2^level]
    """
    B, C, L = x.shape
    assert L % (2**level) == 0
    x = x.view(B, C, -1, 2**level).mean(dim=-1)
    scale = x.new_tensor(2**(0.5 * level))
    return x * scale
