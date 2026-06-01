"""
Reconstruct a large batch of Lagrangian trajectories using training-free 
guidance with an unconditional diffusion model and partial measurements. 
The results are saved as a large numpy array for further evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from palette_diffusion.palette_datasets import load_palette_data
from tfg_diffusion.script_util import (
    tfg_model_and_diffusion_defaults, guidance_defaults,
    tfg_create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = tfg_create_model_and_diffusion(
        **args_to_dict(args, tfg_model_and_diffusion_defaults().keys()),
        task=args.task, guidance_kwargs=args_to_dict(args, guidance_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    data = load_palette_data(
        mask_mode=args.mask_mode,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
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

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        _, guider_kwargs = next(data)
        guider_kwargs["mask"] = 1.0 - guider_kwargs["mask"]
        guider_kwargs = {k: v.to(dist_util.dev()) for k, v in guider_kwargs.items()}

        sample = diffusion.sample(
            model,
            guider_kwargs["mask"].shape,  # (args.batch_size, args.in_channels, args.image_size),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            guider_kwargs=guider_kwargs,
        )
        sample = sample.clamp(-1, 1)
        sample = sample.permute(0, 2, 1)
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
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        seed=0,
    )
    defaults.update(tfg_model_and_diffusion_defaults())
    defaults.update(guidance_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
