"""
Use DDIM reverse and forward processes to encode and reconstruct a large batch of 
Lagrangian trajectories or 2d/3d rotating turbulence fields. The latent noises and 
reconstructions are saved as numpy arrays.
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

    logger.log("encoding and reconstructing...")
    all_noises = []
    all_images = []
    all_labels = []
    xstart = th.from_numpy(np.swapaxes(np.load(args.xstarts_path), 1, 2)
        ).to(dtype=th.float32, device=dist_util.dev())
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
        shape = (args.batch_size, args.in_channels, 
            args.depth_size, *((args.dims-1)*[args.image_size]))
        noise = diffusion.ddim_reverse_sample_loop(
            model,
            shape,
            xstart=xstart,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = diffusion.ddim_sample_loop(
            model,
            shape,
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = sample.clamp(-1, 1)
        #sample[:, -1] = sample[:, -1].clamp(-1, 1)
        #sample = sample.permute(0, 2, 1)
        #sample = sample.permute(0, 1, 3, 2)
        sample = sample.permute(0, *(2+np.arange(args.dims)), 1)
        sample = sample.contiguous()
        noise = noise.permute(0, 2, 1)
        noise = noise.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_noises = [th.zeros_like(noise) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_noises, noise)
        all_noises.extend([noise.cpu().numpy() for noise in gathered_noises])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(
            f"created {len(all_images) * args.batch_size} noises and reconstructions")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    noise_arr = np.concatenate(all_noises, axis=0)
    noise_arr = noise_arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), 
            f"reconstructions_noises_{shape_str}-seed{args.seed:03d}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, noise_arr, label_arr)
        else:
            np.savez(out_path, arr, noise_arr)

    dist.barrier()
    logger.log("encoding and reconstructing complete")


def create_argparser():
    defaults = dict(
        depth_size=0,
        use_unet3d=False,  # use 3D-supported UNetModel or not
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=True,
        model_path="",
        class_label=0,
        seed=0,
        xstarts_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
