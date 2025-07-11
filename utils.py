import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.physics import ricker_wavelet, forward_wave_propagation

class SyntheticFWIDataset(Dataset):
    def __init__(self, n_samples=100, nx=64, nz=64, nt=100, dt=0.004, dx=10.0, freq=10.0):
        self.n_samples = n_samples
        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.dt = dt
        self.dx = dx
        self.freq = freq
        self.data = []
        self.labels = []
        for _ in range(n_samples):
            velocity = self.generate_velocity_model()  # [1, nx, nz]
            source = ricker_wavelet(freq, nt, dt)
            seismic = forward_wave_propagation(velocity.unsqueeze(0), source, nt, dt, dx)  # [1, nt, nx, nz]
            # Store as [nt, nx, nz] and [1, nx, nz]
            self.data.append(seismic.squeeze(0).detach().cpu().numpy())
            self.labels.append(velocity.detach().cpu().numpy())
    def generate_velocity_model(self):
        # Simple two-layer model with random interface
        model = np.ones((1, self.nx, self.nz), dtype=np.float32) * 1500
        interface = np.random.randint(self.nz//3, 2*self.nz//3)
        model[:, :, interface:] = 2500
        return torch.tensor(model)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        # Ensure correct shapes
        data = torch.tensor(self.data[idx], dtype=torch.float32)  # [nt, nx, nz]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # [1, nx, nz]
        # Debug prints
        # print(f"getitem data shape: {data.shape}, label shape: {label.shape}")
        return data, label

def get_dataloaders(batch_size=4, n_train=80, n_val=20, **kwargs):
    train_set = SyntheticFWIDataset(n_samples=n_train, **kwargs)
    val_set = SyntheticFWIDataset(n_samples=n_val, **kwargs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader 