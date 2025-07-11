import torch
from models.unet import UNet
from utils import get_dataloaders
import matplotlib.pyplot as plt

def evaluate(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (seismic, velocity) in enumerate(val_loader):
            seismic = seismic.unsqueeze(1).to(device)
            velocity = velocity.to(device)
            input_data = seismic.mean(dim=2)
            pred = model(input_data)
            # Plot first sample
            if i == 0:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(velocity[0,0].cpu(), cmap='seismic')
                axs[0].set_title('True Velocity')
                axs[1].imshow(pred[0,0].cpu(), cmap='seismic')
                axs[1].set_title('Predicted Velocity')
                axs[2].imshow(input_data[0,0].cpu(), cmap='gray')
                axs[2].set_title('Input (Seismic Mean)')
                plt.tight_layout()
                plt.show()
                break

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader = get_dataloaders(batch_size=1, n_train=10, n_val=2, nx=32, nz=32, nt=30)
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('fwi_hybrid_unet.pth', map_location=device))
    evaluate(model, val_loader, device)

if __name__ == '__main__':
    main() 