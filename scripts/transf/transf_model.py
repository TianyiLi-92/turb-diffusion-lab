"""
Print the model summary of DiT being used.
"""

import argparse

import torch
from torchsummary import summary

from guided_diffusion import dist_util, logger
from transformer_diffusion.script_util import(
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()
    if args.num_classes == 0:
        args.num_classes = None

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    input_size = [(args.in_channels, args.image_size), (), ()]
    dtypes = [torch.FloatTensor, torch.FloatTensor, torch.IntTensor]
    summary(model, input_size, dtypes=dtypes)


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
