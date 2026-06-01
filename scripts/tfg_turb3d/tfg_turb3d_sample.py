"""
Reconstruct a large batch of 2d/3d rotating turbulence fields from partial measurements 
using training-free guidance on an unconditional diffusion model, with physics 
constraints applied as specified by --constraint_name (options: "no", "2d", or "3d").
The reconstructed results are saved as a large numpy array for further evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from turb3d_diffusion.turb_datasets import load_data
from turb3d_diffusion.script_util import (
    turb3d_model_and_diffusion_defaults,
)
from tfg_diffusion.script_util import guidance_defaults
from tfg_turb3d_diffusion.script_util import (
    tfg_turb3d_create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    assert 0 <= args.depth_size <= args.image_size
    if args.depth_size == 0:
        args.depth_size = None

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = tfg_turb3d_create_model_and_diffusion(
        **args_to_dict(args, turb3d_model_and_diffusion_defaults().keys()),
        task=args.task,
        guidance_kwargs=args_to_dict(args, guidance_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        depth_size=args.depth_size,
        deterministic=True,
    )

    logger.log("sampling...")
    all_images = []

    # Set the initial noise for sampling.
    #noise = th.zeros(
    # noise = th.ones(
    #     (args.batch_size, args.in_channels, args.image_size),
    #     dtype=th.float32,
    #     device=dist_util.dev()
    # ) * 2
    # noise = th.from_numpy(
    #     np.load('../velocity_module-IS64-NC128-NRB3-DS4000-NScosine-LR1e-4-BS256-sample/fixed_noise_64x1x64x64.npy')
    # ).to(dtype=th.float32, device=dist_util.dev())

    # Set global random seeds for all GPUs.
    seed = args.seed*4 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)

    shape = (args.batch_size, args.in_channels, 
        args.depth_size, *((args.dims-1)*[args.image_size]))
    mask = th.from_numpy(measurement_mask(shape, args.mask_mode)
        ).to(dtype=th.float32, device=dist_util.dev())
    guider_kwargs = {"mask": mask}

    while len(all_images) * args.batch_size < args.num_samples:
        batch, model_kwargs = next(data)
        guider_kwargs["cond"] = batch.to(dist_util.dev())

        sample = diffusion.sample(
            model,
            shape,
            #noise=noise,
            clip_denoised=args.clip_denoised,
            denoised_fn=diffusion.constraint_fn,
            model_kwargs=model_kwargs,
            guider_kwargs=guider_kwargs,
        )
        sample = sample.clamp(-1, 1)
        sample = sample.permute(0, *(2+np.arange(args.dims)), 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(
            logger.get_dir(), f"samples_{shape_str}-seed{args.seed:03d}.npz"
        )
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        task="inpainting",
        mask_mode="",
        dataset_path="",
        dataset_name="",
        depth_size=0,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        seed=0,
    )
    defaults.update(turb3d_model_and_diffusion_defaults())
    defaults.update(guidance_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def measurement_mask(shape, mask_mode):
    mask = np.zeros(shape)  # [B, C, Nz, Nx, Ny]
    if mask_mode.startswith('slicex'):
        idx0, idx1 = map(int, mask_mode[6:].split('_'))
        mask[:, :, :, idx0:idx1] = 1.0
    elif mask_mode.startswith('slicez'):
        idz0, idz1 = map(int, mask_mode[6:].split('_'))
        mask[:, :, idz0:idz1] = 1.0
    else:
        raise NotImplementedError(f"unsupported mask mode: {mask_mode}")
    return mask


if __name__ == "__main__":
    main()
