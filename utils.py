import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.physics import ricker_wavelet, forward_wave_propagation

# Velocity range of the synthetic models, used to normalize network inputs/outputs.
V_MIN = 1500.0
V_MAX = 2500.0

def normalize_velocity(v):
    return (v - V_MIN) / (V_MAX - V_MIN)

def denormalize_velocity(v_norm):
    return V_MIN + v_norm * (V_MAX - V_MIN)

class SyntheticFWIDataset(Dataset):
    # Stability of the 2D 5-point scheme requires v_max * dt / dx <= 1/sqrt(2),
    # so with v_max=2500 and dx=10 we need dt <= ~0.0028.
    def __init__(self, n_samples=100, nx=64, nz=64, nt=100, dt=0.002, dx=10.0, freq=10.0):
        self.n_samples = n_samples
        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.dt = dt
        self.dx = dx
        self.freq = freq
        self.data = []
        self.labels = []
        with torch.no_grad():
            for _ in range(n_samples):
                velocity = self.generate_velocity_model()  # [1, nx, nz]
                source = ricker_wavelet(freq, nt, dt)
                seismic = forward_wave_propagation(velocity.unsqueeze(0), source, nt, dt, dx)  # [1, nt, nx, nz]
                self.data.append(seismic.squeeze(0))   # [nt, nx, nz]
                self.labels.append(velocity)           # [1, nx, nz]
    def generate_velocity_model(self):
        # Simple two-layer model with random interface
        model = np.ones((1, self.nx, self.nz), dtype=np.float32) * V_MIN
        interface = np.random.randint(self.nz//3, 2*self.nz//3)
        model[:, :, interface:] = V_MAX
        return torch.tensor(model)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # [nt, nx, nz], [1, nx, nz]

def get_dataloaders(batch_size=4, n_train=80, n_val=20, **kwargs):
    train_set = SyntheticFWIDataset(n_samples=n_train, **kwargs)
    val_set = SyntheticFWIDataset(n_samples=n_val, **kwargs)
    # n_train=0 lets evaluation skip generating a training set it won't use
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) if n_train > 0 else None
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
