from guided_diffusion.script_util import create_model as create_model_nd
from guided_diffusion import gaussian_diffusion as gd
from .tfg_turb3d_diffusion import (
    BaseGuidanceTurb3dDiffusion,
    DPSGuidanceTurb3dDiffusion,
    TFGGuidanceTurb3dDiffusion,
)


def tfg_turb3d_create_model_and_diffusion(
    use_unet3d,
    dims,
    image_size,
    in_channels,
    class_cond,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    constraint_name,
    diffusion_steps,
    sigma_small,
    noise_schedule,
    predict_xstart,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    task,
    guidance_kwargs,
):
    create_model = create_model_nd if not use_unet3d else create_model_3d
    model = create_model(
        dims,
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=False,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_tfg_turb3d_diffusion(
        task=task,
        guidance_kwargs=guidance_kwargs,
        constraint_name=constraint_name,
        steps=diffusion_steps,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
    )
    return model, diffusion


def create_tfg_turb3d_diffusion(
    *,
    task="",
    guidance_kwargs=None,
    constraint_name="no",
    steps=1000,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    rescale_timesteps=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    model_mean_type = (
        gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
    )
    model_var_type = (
        gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL
    )
    loss_type = gd.LossType.MSE
    guidance_name = guidance_kwargs["guidance_name"]
    if guidance_name == "no":
        GuidanceTurb3dDiffusion = BaseGuidanceTurb3dDiffusion
    elif guidance_name == "dps":
        GuidanceTurb3dDiffusion = DPSGuidanceTurb3dDiffusion
    elif guidance_name == "tfg":
        GuidanceTurb3dDiffusion = TFGGuidanceTurb3dDiffusion
    else:
        raise NotImplementedError
    return GuidanceTurb3dDiffusion(
        task=task,
        guidance_kwargs=guidance_kwargs,
        constraint_name=constraint_name,
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
