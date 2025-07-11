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
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## File Structure
- `models/unet.py`: U-Net model
- `models/physics.py`: Physics-based forward operator
- `utils.py`: Data loading and synthetic data generation
- `train.py`: Training script
- `evaluate.py`: Evaluation script

## Notes
- This starter uses synthetic data for demonstration. Replace with real data as needed. 