# turb-diffusion-lab

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
