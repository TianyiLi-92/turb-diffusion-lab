"""
Generate a large batch of Lagrangian trajectories from a transformer model and save them
as a large numpy array. This can be used to produce samples for statistical evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from transformer_diffusion.script_util import(
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    if args.num_classes == 0:
        args.num_classes = None
        class_cond = False
        use_cfg = False
    else:
        class_cond = True
        assert 0 <= args.class_label < args.num_classes
        use_cfg = True if args.cfg_scale > 1.0 else False

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
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
        noise, model_kwargs = None, {}
        if class_cond:
            # classes = th.randint(
            #     low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.full(
                size=(args.batch_size,),
                fill_value=args.class_label,
                dtype=th.int64,
                device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            if use_cfg:
                noise = th.randn(
                    args.batch_size, args.in_channels, args.image_size,
                    device=dist_util.dev()
                )
                noise = th.cat([noise, noise], 0)
                classes_null = th.full(
                    size=(args.batch_size,),
                    fill_value=args.num_classes,
                    dtype=th.int64,
                    device=dist_util.dev()
                )
                model_kwargs["y"] = th.cat([classes, classes_null], 0)
                model_kwargs["cfg_scale"] = args.cfg_scale
        else:
            classes = th.full(
                size=(args.batch_size,),
                fill_value=-1,
                dtype=th.int64,
                device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        #sample_fn = diffusion.p_sample_loop_history
        sample = sample_fn(
            model if not use_cfg else model.forward_with_cfg,
            noise.shape if noise is not None else
            (args.batch_size, args.in_channels, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
        )
        if use_cfg:
            sample, _ = sample.chunk(2, dim=0)
        sample = sample.clamp(-1, 1)
        #sample[:, -1] = sample[:, -1].clamp(-1, 1)
        sample = sample.permute(0, 2, 1)
        #sample = sample.permute(0, 1, 3, 2)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(
            logger.get_dir(),
            f"samples_{shape_str}-seed{args.seed:03d}.npz" if not use_cfg else
            f"samples_{shape_str}-cfg_scale{args.cfg_scale}-seed{args.seed:03d}.npz"
        )
        logger.log(f"saving to {out_path}")
        if class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        class_label=0,
        cfg_scale=1.0,
        seed=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
