import torch
from models.unet import UNet
from utils import get_dataloaders, denormalize_velocity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate(model, val_loader, device):
    model.eval()
    total_mse, n = 0.0, 0
    with torch.no_grad():
        for i, (seismic, velocity) in enumerate(val_loader):
            seismic = seismic.unsqueeze(1).to(device)
            velocity = velocity.to(device)
            input_data = seismic.mean(dim=2)
            pred = denormalize_velocity(torch.sigmoid(model(input_data)))  # m/s
            total_mse += torch.nn.functional.mse_loss(pred, velocity).item() * velocity.shape[0]
            n += velocity.shape[0]
            # Plot first sample
            if i == 0:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                im0 = axs[0].imshow(velocity[0,0].cpu(), cmap='seismic', vmin=1500, vmax=2500)
                axs[0].set_title('True Velocity (m/s)')
                fig.colorbar(im0, ax=axs[0])
                im1 = axs[1].imshow(pred[0,0].cpu(), cmap='seismic', vmin=1500, vmax=2500)
                axs[1].set_title('Predicted Velocity (m/s)')
                fig.colorbar(im1, ax=axs[1])
                axs[2].imshow(input_data[0,0].cpu(), cmap='gray')
                axs[2].set_title('Input (Seismic Mean)')
                plt.tight_layout()
                plt.savefig('evaluation.png', dpi=150)
                print('Saved evaluation.png')
    rmse = (total_mse / n) ** 0.5
    print(f'Validation velocity RMSE: {rmse:.1f} m/s')

def main():
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nt = 30
    dt = 0.002
    dx = 10.0
    freq = 10.0
    _, val_loader = get_dataloaders(batch_size=1, n_train=0, n_val=2, nx=32, nz=32, nt=nt, dt=dt, dx=dx, freq=freq)
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('fwi_hybrid_unet.pth', map_location=device))
    evaluate(model, val_loader, device)

if __name__ == '__main__':
    main()
