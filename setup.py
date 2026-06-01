from setuptools import setup, find_packages

setup(
    name="turb-diffusion-lab",
    packages=find_packages(include=[
        "guided_diffusion",
        "continuous_diffusion",
        "transformer_diffusion",
        "palette_diffusion",
        "tfg_diffusion",
        "turb3d_diffusion",
        "tfg_turb3d_diffusion",
        "utils",
    ]),
    install_requires=[
        "blobfile>=1.0.5",
        "tqdm",
        "h5py",
        "timm",
    ],
)
