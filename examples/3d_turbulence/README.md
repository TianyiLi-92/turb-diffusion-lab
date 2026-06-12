# 3D turbulence

Minimal examples for training and sampling a physics-constrained diffusion model (PCDM) for 3D turbulent velocity fields.

For detailed configurations and additional model variants, please refer to the original repository:
https://github.com/SmartTURB/pcdm-turb3d

By default, the scripts run on one GPU. To use multiple GPUs, set `NUM_GPUS`, for example:

```bash
NUM_GPUS=2 bash examples/3d_turbulence/train_pcdm.sh
```

## PCDM training

```bash
bash examples/3d_turbulence/train_pcdm.sh
```

## PCDM sampling

Create the checkpoint directory and download the pretrained `PCDM-Fourier` checkpoint:

```bash
mkdir -p examples/3d_turbulence/checkpoints/pcdm_fourier

curl -L "https://www.openaccessrepository.it/records/0tb37-k2367/files/pcdm_fourier-ema_0.9999_800000.pt?download=1" \
    -o examples/3d_turbulence/checkpoints/pcdm_fourier/ema_0.9999_800000.pt
```

Then run:

```bash
bash examples/3d_turbulence/sample_pcdm.sh
```

Additional examples will be added progressively.
