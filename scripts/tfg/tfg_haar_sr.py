"""
Performs super-resolution of a large batch of Lagrangian trajectories using 
a training-free guidance framework. An unconditional diffusion model is 
guided by Haar wavelet approximation coefficients representing the coarse-scale 
observations. The outputs are saved as a numpy array for further analysis.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.turb_datasets import load_data
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
    data = load_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
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
        batch, model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        guider_kwargs = {}
        guider_kwargs["level"] = args.approx_level
        guider_kwargs["reference"] = batch.to(dist_util.dev())

        sample = diffusion.sample(
            model,
            guider_kwargs["reference"].shape,  # (args.batch_size, args.in_channels, args.image_size),
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
        task="haar_sr",
        approx_level=0,  # Haar approximation level; lower = finer, higher = coarser
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
