# Lagrangian turbulence generation

By default, the scripts run on one GPU. To use multiple GPUs, set `NUM_GPUS`, for example:

```bash
NUM_GPUS=2 bash examples/lagrangian_generation/train_unet.sh
```

## U-Net training

```bash
bash examples/lagrangian_generation/train_unet.sh
```

## U-Net sampling

Create the checkpoint directory and download the pretrained `DM-3c` checkpoint:

```bash
mkdir -p examples/lagrangian_generation/checkpoints/unet/unconditional

curl -L "https://www.dropbox.com/scl/fi/o7aun6o7lfk99eikds4c2/ema_0.9999_400000.pt?rlkey=mkxaxs0kw4ighb330a3yca0mo&dl=1" \
    -o examples/lagrangian_generation/checkpoints/unet/unconditional/ema_0.9999_400000.pt
```

Then run:

```bash
bash examples/lagrangian_generation/sample_unet.sh
```

Additional examples will be added progressively.
