import torch
import torch.nn.functional as F

def ricker_wavelet(f, length, dt):
    t = torch.arange(-length//2, (length+1)//2) * dt
    y = (1.0 - 2.0 * (torch.pi**2) * (f**2) * (t**2)) * torch.exp(-(torch.pi**2) * (f**2) * (t**2))
    return y

def forward_wave_propagation(velocity, source, nt, dt, dx):
    """
    Simple 2D acoustic wave propagation using finite differences.
    velocity: [batch, 1, nx, nz]
    source: [nt] (source time function)
    Returns: [batch, nt, nx, nz] (seismic data at each time step)
    """
    batch, _, nx, nz = velocity.shape
    p = torch.zeros((batch, nx, nz), device=velocity.device)
    p_prev = torch.zeros_like(p)
    data = []
    for it in range(nt):
        lap = (
            -4 * p
            + torch.roll(p, shifts=1, dims=1)
            + torch.roll(p, shifts=-1, dims=1)
            + torch.roll(p, shifts=1, dims=2)
            + torch.roll(p, shifts=-1, dims=2)
        ) / (dx**2)
        p_new = 2*p - p_prev + (velocity[:,0]**2) * (dt**2) * lap
        # Add source at center for each batch
        p_new[:, nx//2, nz//2] += source[it]
        data.append(p_new.unsqueeze(1))
        p_prev = p
        p = p_new
    return torch.cat(data, dim=1)  # [batch, nt, nx, nz] 