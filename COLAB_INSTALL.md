# Colab install cell

Use this cell at the top of your Colab notebook. It installs a compatible PyTorch, prebuilt PyTorch3D and xformers wheel, and other dependencies without compiling from source.

```bash
# 1) Install a PyTorch wheel compatible with Colab (adjust CUDA version if needed)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2) Prebuilt xformers wheel (choose matching torch+cuda) - replace URL if needed
# 2) Optional: try to install prebuilt xformers wheel ONLY (do NOT fall back to source build)
# If no binary wheel exists for your Python/Torch/CUDA combination this will skip silently.
pip install --upgrade pip
pip install --only-binary=:all: xformers -f https://download.pytorch.org/whl/torch_stable.html || echo 'No prebuilt xformers wheel available for this environment; skipping.'

# 3) Optional: try to install prebuilt pytorch3d wheel ONLY
# Replace the -f index URL if you have a different wheel source. This will NOT compile from source.
pip install --only-binary=:all: pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3-none-any/ || echo 'No prebuilt pytorch3d wheel available for this environment; skipping.'

# 4) Optional automated attempt (recommended): run the helper that only tries binary wheels
python tools/install_optional_wheels.py

# 4) Install other Python deps from requirements
pip install -r requirements.txt

# 5) Optional: GPU memory optimizations
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
```

Notes:
- If the prebuilt wheels above are unavailable for your current CUDA/PyTorch combination, search for official prebuilt wheels for `pytorch3d` and `xformers` matching the exact torch+cuda.
- Do NOT attempt to compile `pytorch3d` or `xformers` from source in Colab: it is slow and often fails.
