# Lagrangian signal reconstruction

By default, the scripts run on one GPU. To use multiple GPUs, set `NUM_GPUS`, for example:

```bash
NUM_GPUS=2 bash examples/lagrangian_reconstruction/train_prior.sh
```

## Prior training for DPS

```bash
bash examples/lagrangian_reconstruction/train_prior.sh
```

## DPS reconstruction

This example uses a small training subset for demonstration only; use separate validation and test sets for formal evaluation.

Create the checkpoint directory and download the pretrained unconditional prior checkpoint:

```bash
mkdir -p examples/lagrangian_reconstruction/checkpoints/prior

curl -L "https://www.dropbox.com/scl/fi/3zt6aainke45cinfy6at4/ema_0.9999_250000.pt?rlkey=n1p3c4m8mparh6u92327nv26k&dl=1" \
    -o examples/lagrangian_reconstruction/checkpoints/prior/ema_0.9999_250000.pt
```

Then run:

```bash
bash examples/lagrangian_reconstruction/reconstruct_dps.sh
```

For multi-GPU reconstruction, use `utils.parallel_utils.reverse_gpu_parallel_indices` to restore the correspondence between generated samples and the original dataset indices.

Additional examples will be added progressively.
