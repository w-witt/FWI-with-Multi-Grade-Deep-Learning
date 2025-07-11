import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from models.physics import forward_wave_propagation, ricker_wavelet
from utils import get_dataloaders
from tqdm import tqdm

def hybrid_loss(pred, target, observed_seismic, velocity_shape, nt, dt, dx, freq):
    # Data loss (MSE between predicted and true velocity)
    data_loss = nn.functional.mse_loss(pred, target)
    # Physics loss (MSE between observed and simulated seismic)
    batch = pred.shape[0]
    physics_loss = 0.0
    for i in range(batch):
        pred_seismic = forward_wave_propagation(
            pred[i].unsqueeze(0),
            ricker_wavelet(freq, nt, dt),
            nt, dt, dx
        )
        physics_loss += nn.functional.mse_loss(pred_seismic.squeeze(0), observed_seismic[i])
    physics_loss /= batch
    return data_loss + physics_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_dataloaders(batch_size=2, n_train=10, n_val=2, nx=32, nz=32, nt=30)
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 2
    nt = 30
    dt = 0.004
    dx = 10.0
    freq = 10.0
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for seismic, velocity in pbar:
            seismic = seismic.unsqueeze(1).to(device)  # [B, 1, nt, nx, nz]
            velocity = velocity.to(device)
            # Collapse time into channel for U-Net input
            input_data = seismic.mean(dim=2)  # [B, 1, nx, nz]
            pred = model(input_data)
            loss = hybrid_loss(pred, velocity, seismic.squeeze(1), velocity.shape, nt, dt, dx, freq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1} complete.")
    torch.save(model.state_dict(), 'fwi_hybrid_unet.pth')

if __name__ == '__main__':
    main() 