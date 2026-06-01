from guided_diffusion.script_util import diffusion_defaults
from .models import DiT_models
from continuous_diffusion.script_util import continuous_create_gaussian_diffusion


def model_and_diffusion_defaults():
    res = dict(
        model="DiT-S/8",
        image_size=2000,
        in_channels=3,
        num_classes=3,
        learn_sigma=False,
    )
    res["use_continuous_diffusion"] = False
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    model,
    image_size,
    in_channels,
    num_classes,
    learn_sigma,
    use_continuous_diffusion,
    diffusion_steps,
    noise_schedule,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    timestep_respacing,
):
    model = DiT_models[model](
        input_size=image_size,
        in_channels=in_channels,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
    )
    diffusion = continuous_create_gaussian_diffusion(
        use_continuous_diffusion=use_continuous_diffusion,
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion
