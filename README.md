# Baseline Hybrid FWI Model

This project provides a baseline hybrid model for Full Waveform Inversion (FWI), combining physics-based and machine learning approaches to reconstruct subsurface images from seismic data.

## Features
- Physics-based 2D wave equation forward operator
- U-Net-based neural network for inversion
- Hybrid loss: combines data and physics consistency
- Synthetic data generation for demonstration

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training:
   ```bash
   python train.py
   ```
3. Evaluate the model (prints validation RMSE and saves `evaluation.png`):
   ```bash
   python evaluate.py
   ```

## File Structure
- `models/unet.py`: U-Net model
- `models/physics.py`: Physics-based forward operator
- `utils.py`: Data loading and synthetic data generation
- `train.py`: Training script
- `evaluate.py`: Evaluation script
- `scripts/download_kimberlina3d_sample.py`: Download a small 3D Kimberlina (OpenFWI Kimberlina-V1) sample from DOE/NETL EDX

## Getting the 3D Kimberlina (OpenFWI) data

The 3D dataset referenced on the [OpenFWI data page](https://openfwi-lanl.github.io/docs/data.html)
is hosted on DOE/NETL's [Energy Data eXchange](https://edx.netl.doe.gov/group/kimberlina-geophysical-data),
whose web UI asks for a login — that is why the download links appear broken.
The files themselves are public, though, and this repo ships a script that
downloads a paired sample (velocity model + shot gathers) directly, no account
needed. It uses HTTP range requests to pull single files out of the multi-GB
archives, so a sample costs ~200 MB instead of ~10 GB:

```bash
# one velocity model (400x400x350) + 3 shot gathers (40x40x5001) as .npy
python scripts/download_kimberlina3d_sample.py

# choose a different simulation year / spatial cut / shots
python scripts/download_kimberlina3d_sample.py --year 95 --cut 7 --shots 1 13 25

# see what's available (33 years x ~63 cuts x 25 shots)
python scripts/download_kimberlina3d_sample.py --list
```

Data format: velocity models are float32 `(400, 400, 350)` with 10 m cells;
each shot gather is float32 `(40, 40, 5001)` — a 40x40 surface geophone grid at
100 m spacing, 5001 time samples, dt = 1 ms, 25 shots per cut.

The data is CC BY-NC-SA 4.0. If you use it, cite OpenFWI (Deng et al., NeurIPS
2022) and the Kimberlina 1.2 dataset (Alumbaugh et al., Geoscience Data
Journal, 2024).

## Notes
- This starter uses synthetic data for demonstration. Replace with real data as needed. 