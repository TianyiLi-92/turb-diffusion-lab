# turb-diffusion-lab

This repository consolidates the main code developed across several projects on diffusion-model-based generation, reconstruction, and physics-constrained modeling of Lagrangian and Eulerian turbulent data. It is intended as a common foundation for future research and development. For detailed usage instructions, datasets, and paper-specific reproduction settings, please refer to the original repositories listed below.

## Projects

### Lagrangian turbulence generation

Generation of synthetic Lagrangian trajectories using DDPM and DDIM with U-Net and transformer-based DiT models. Class-conditional generation for different particle types is also supported.

Original repositories:  
https://github.com/SmartTURB/diffusion-lagr  
https://github.com/SmartTURB/transf-DM-lagr

### Lagrangian signal reconstruction

Reconstruction of gappy Lagrangian trajectories using conditional diffusion models or diffusion posterior sampling (DPS) with a pretrained generative diffusion model prior.

Original repository: https://github.com/SmartTURB/C-DM-lagr

### 3D turbulence generation and reconstruction

Generation of 3D turbulent velocity fields using physics-constrained diffusion models, including DPS-based reconstruction from partial observations.

Original repository: https://github.com/SmartTURB/pcdm-turb3d

### Related legacy repositories

Earlier implementations for the reconstruction of incomplete 2D turbulent snapshots are available at:

- https://github.com/SmartTURB/repaint-turb
- https://github.com/SmartTURB/palette-turb

Future development of related functionality is recommended within the unified framework of this repository.

## Installation

<details open>
<summary><strong>Using <code>venv</code> + <code>pip</code> (recommended)</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# Optional: load MPI first on systems that provide it as a module
module load mpi

pip install mpi4py

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install git+https://github.com/sksq96/pytorch-summary.git

pip install -e .
```
</details>

<details>
<summary><strong>Using <code>conda</code></strong></summary>

```bash
conda create -n turb-diffusion-lab python=3.7 mpi4py openmpi
conda activate turb-diffusion-lab

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

pip install git+https://github.com/sksq96/pytorch-summary.git

pip install -e .
```
</details>

## Examples

Minimal examples for the main workflows will be added progressively.
