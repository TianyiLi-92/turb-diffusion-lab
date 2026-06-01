"""
Generate a large batch of 2d/3d rotating turbulence fields from a model and save them 
as a large numpy array for statistical evaluation. A physics constraint function is 
applied during each backward sampling step, passed in via the denoised_fn argument.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from continuous_diffusion.script_util import (
    continuous_model_and_diffusion_defaults as model_and_diffusion_defaults,
    create_model_and_diffusion as create_model_and_diffusion_nd,
)
from turb3d_diffusion.script_util import (
    create_model_and_diffusion as create_model_and_diffusion_3d,
)
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    assert 0 <= args.depth_size <= args.image_size
    if args.depth_size == 0:
        args.depth_size = args.image_size
    if args.class_cond:
        assert 0 <= args.class_label < NUM_CLASSES

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    create_model_and_diffusion = (
        create_model_and_diffusion_nd if not args.use_unet3d
        else create_model_and_diffusion_3d
    )
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    #noise = th.zeros(
    # noise = th.ones(
    #     (args.batch_size, args.in_channels, args.image_size),
    #     dtype=th.float32,
    #     device=dist_util.dev()
    # ) * 2
    # noise = th.from_numpy(
    #     np.load('../velocity_module-IS64-NC128-NRB3-DS4000-NScosine-LR1e-4-BS256-sample/fixed_noise_64x1x64x64.npy')
    # ).to(dtype=th.float32, device=dist_util.dev())
    import os
    seed = args.seed*4 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.full(
                size=(args.batch_size,),
                fill_value=args.class_label,
                dtype=th.int64,
                device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        #sample_fn = diffusion.p_sample_loop_history
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, 
            args.depth_size, *((args.dims-1)*[args.image_size])),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            #denoised_fn=constraint_fn_3d,
            denoised_fn=incompressible_projection,
            model_kwargs=model_kwargs,
        )
        sample = sample.clamp(-1, 1)
        #sample[:, -1] = sample[:, -1].clamp(-1, 1)
        #sample = sample.permute(0, 2, 1)
        #sample = sample.permute(0, 1, 3, 2)
        sample = sample.permute(0, *(2+np.arange(args.dims)), 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(
            logger.get_dir(), f"samples_{shape_str}-seed{args.seed:03d}.npz"
        )
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        depth_size=0,
        use_unet3d=False,  # use 3D-supported UNetModel or not
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        class_label=0,
        seed=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def constraint_fn_3d(x):
    x = adjust_mean(x, 0, (-2, -1), -0.06546515)
    x = adjust_mean(x, 1, (-3, -1), -0.02231663)
    x = adjust_mean(x, 2, (-3, -2), -0.01817745)
    return x


def adjust_mean(x, comp, axis, target_mean=0.0):
    """
    Adjust the mean of the specified velocity component over the given 
    axes to a target value.

    :param x: Tensor of shape [B, C, D, H, W], the velocity field.
    :param comp: int, the index of the velocity component to adjust.
    :param axis: tuple of ints, the axes over which to compute the mean.
    :param target_mean: float, the desired mean value after adjustment.

    Examples:
    - For any comp and axis=(-3, -2, -1): set mean over the volume.
    - For comp=0 and axis=(-2, -1): set mean of ux over the yz-plane, 
      as required by incompressibility and volume-averaged zero mean.
    """
    current_mean = x[:, comp].mean(dim=axis, keepdim=True)
    x_comp = x[:, comp] - current_mean + target_mean
    x_comp = x_comp.unsqueeze(1)
    x = x.index_copy(1, th.tensor([comp], device=x.device), x_comp)
    return x


def incompressible_projection(x):
    """
    Project the velocity field onto a divergence-free space in Fourier space 
    and remove the zero mode (mean component).

    :param x: Tensor of shape [B, C, D, H, W], the velocity field.
    :return: Tensor of the same shape after the projection.
    """
    B, C, D, H, W = x.shape
    assert C == 3, "Velocity field must have 3 components."
    assert D == H == W, "Only cubic domains are supported."
    assert W % 2 == 0, "Grid size must be even for rFFT."

    # Range mapping constants for velocity components (specific to Rot64_diffusion.h5)
    rx0 = th.tensor([-10.823559, -13.279927, -13.323344], 
        dtype=x.dtype, device=x.device).view(1, C, 1, 1, 1)
    rx1 = th.tensor([12.339963, 13.886183, 13.81668], 
        dtype=x.dtype, device=x.device).view(1, C, 1, 1, 1)

    # Compute wave numbers for Fourier space projection
    kx = th.fft.fftfreq(D, device=x.device).view(1, 1, D, 1, 1) * D
    ky = th.fft.fftfreq(H, device=x.device).view(1, 1, 1, H, 1) * H
    kz = th.fft.rfftfreq(W, device=x.device).view(1, 1, 1, 1, W // 2 + 1) * W

    shape = (1, 1, D, H, W // 2 + 1)
    kx, ky, kz = kx.expand(*shape), ky.expand(*shape), kz.expand(*shape)

    kv = th.cat((kx, ky, kz), dim=1)  # Wave vector
    k2 = kx**2 + ky**2 + kz**2
    k2_avoid_zero = th.where(k2 == 0, th.ones_like(k2), k2)  # Prevent division by zero

    x = (x+1)*(rx1-rx0)/2 + rx0
    x_spec = th.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

    # Compute divergence-free projection and remove zero mode
    x_spec = x_spec - th.sum(kv * x_spec, dim=1, keepdim=True) * kv / k2_avoid_zero
    x_spec = th.where(k2 > 1e-6, x_spec, th.zeros_like(x_spec))

    x = th.fft.irfftn(x_spec, dim=(-3, -2, -1), norm="ortho")
    x = 2*(x-rx0)/(rx1-rx0) - 1
    return x


if __name__ == "__main__":
    main()
