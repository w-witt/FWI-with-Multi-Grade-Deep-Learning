import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from models.physics import forward_wave_propagation, ricker_wavelet
from utils import get_dataloaders, normalize_velocity, denormalize_velocity
from tqdm import tqdm

def hybrid_loss(pred_norm, target, observed_seismic, source, nt, dt, dx, lambda_physics):
    # Data loss: MSE in normalized velocity space so both terms are O(1)
    data_loss = nn.functional.mse_loss(pred_norm, normalize_velocity(target))
    # Physics loss: re-simulate seismic data from the predicted velocity (batched)
    pred_seismic = forward_wave_propagation(denormalize_velocity(pred_norm), source, nt, dt, dx)
    physics_loss = nn.functional.mse_loss(pred_seismic, observed_seismic)
    return data_loss + lambda_physics * physics_loss

def run_validation(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for seismic, velocity in val_loader:
            seismic = seismic.unsqueeze(1).to(device)
            velocity = velocity.to(device)
            input_data = seismic.mean(dim=2)
            pred_norm = torch.sigmoid(model(input_data))
            total += nn.functional.mse_loss(pred_norm, normalize_velocity(velocity)).item() * velocity.shape[0]
            n += velocity.shape[0]
    return total / n

def main():
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nt = 30
    dt = 0.002  # keeps v_max*dt/dx = 0.5 < 1/sqrt(2), so the FD scheme is stable
    dx = 10.0
    freq = 10.0
    lambda_physics = 1.0
    train_loader, val_loader = get_dataloaders(batch_size=2, n_train=10, n_val=2, nx=32, nz=32, nt=nt, dt=dt, dx=dx, freq=freq)
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 2
    source = ricker_wavelet(freq, nt, dt).to(device)
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for seismic, velocity in pbar:
            seismic = seismic.unsqueeze(1).to(device)  # [B, 1, nt, nx, nz]
            velocity = velocity.to(device)
            # Collapse time into channel for U-Net input
            input_data = seismic.mean(dim=2)  # [B, 1, nx, nz]
            # Sigmoid keeps the prediction in [0,1], i.e. within [V_MIN, V_MAX] after
            # denormalization, which also guarantees the physics simulation stays stable.
            pred_norm = torch.sigmoid(model(input_data))
            loss = hybrid_loss(pred_norm, velocity, seismic.squeeze(1), source, nt, dt, dx, lambda_physics)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        val_mse = run_validation(model, val_loader, device)
        print(f"Epoch {epoch+1} complete. Val MSE (normalized velocity): {val_mse:.6f}")
    torch.save(model.state_dict(), 'fwi_hybrid_unet.pth')

if __name__ == '__main__':
    main()
