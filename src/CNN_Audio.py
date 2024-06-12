import numpy as np
import torch as th
from torch.utils.data import Dataset

class Database(Dataset):
    def __init__(self, X):
        self.X = th.from_numpy(X)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

def make_encoder(latent_dim):
    return th.nn.Sequential(
        th.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=255, stride=2, padding=127),  # (1, 90000) -> (8, 45000) 
        th.nn.ReLU(),
        
        th.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=127, stride=2, padding=63), # (8, 45000) -> (16, 22500) 
        th.nn.ReLU(),
        
        th.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=63, stride=2, padding=31), # (16, 22500) -> (32, 11250) 
        th.nn.ReLU(),
        
        th.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=2, padding=15), # (32, 11250) -> (64, 5625) 
        th.nn.ReLU(),

        th.nn.Flatten(1),                                                                    # (64, 5625) -> (64*5625)
        th.nn.Linear(in_features=64*5625, out_features=1024),                                    # (64*5625) -> (1024)
        th.nn.ReLU()
    )

def make_decoder(latent_dim, in_dim):
    return th.nn.Sequential(
        th.nn.Linear(latent_dim, 1024),                                                                # (latent_dim) -> (1024)
        th.nn.ReLU(),
        
        th.nn.Linear(1024, 5625 * 64),                                                           # (latent_dim) -> (5625)
        th.nn.ReLU(),
        th.nn.Unflatten(1, (64, 5625)),                                                                 # (5625) -> (1, 5625)

        th.nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=256, stride=2, padding=127),  # (1, 5625) -> (1, 11250) 
        th.nn.ReLU(),

        th.nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=128, stride=2, padding=63),   # (1, 11250) -> (1, 22500) 
        th.nn.ReLU(),

        th.nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=64, stride=2, padding=31),   # (1, 22500) -> (1, 45000) 
        th.nn.ReLU(),

        th.nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=32, stride=2, padding=15),   # (1, 45000) -> (1, 90000) 
    )

class VAE(th.nn.Module):
    def __init__(self, latent_dim, in_dim):
        super(VAE, self).__init__()

        self.encoder = make_encoder(latent_dim)

        self.mu = th.nn.Linear(1024, latent_dim)
        self.log_var = th.nn.Linear(1024, latent_dim)

        self.decoder = make_decoder(latent_dim, in_dim)

    def forward(self, x):
        x = self.encoder(x)

        mu, log_var = self.mu(x), self.log_var(x)

        z = mu + th.exp(0.5 * log_var) * th.randn_like(mu)

        out = self.decoder(z)

        return out, mu, log_var