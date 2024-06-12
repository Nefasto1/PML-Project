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

class Resnet1D(th.nn.Module):
    def __init__(self, ch_in, ch_hid, kernel_size):
        super(Resnet1D, self).__init__()

        self.conv1 = th.nn.Conv1d(in_channels=ch_in, out_channels=ch_hid, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.conv2 = th.nn.Conv1d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.conv3 = th.nn.Conv1d(in_channels=ch_hid, out_channels=ch_in, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        # self.norm1 = th.nn.InstanceNorm1d(ch_out)
        # self.norm2 = th.nn.InstanceNorm1d(ch_in)
        self.norm1 = th.nn.BatchNorm1d(ch_hid)
        self.norm2 = th.nn.BatchNorm1d(ch_hid)
        self.norm3 = th.nn.BatchNorm1d(ch_in)

        self.leaky = th.nn.LeakyReLU(0.4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky( self.norm1(out) )

        out = self.conv2(out)
        out = self.leaky( self.norm2(out) )

        out = self.conv3(out)
        out = x + self.leaky( self.norm3(out) )

        return th.relu(out)

class ResnetTransposed1D(th.nn.Module):
    def __init__(self, ch_in, ch_hid, kernel_size):
        super(ResnetTransposed1D, self).__init__()

        self.conv1 = th.nn.ConvTranspose1d(in_channels=ch_in, out_channels=ch_hid, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.conv2 = th.nn.ConvTranspose1d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.conv3 = th.nn.ConvTranspose1d(in_channels=ch_hid, out_channels=ch_in, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.norm1 = th.nn.InstanceNorm1d(ch_hid)
        self.norm2 = th.nn.InstanceNorm1d(ch_hid)
        self.norm3 = th.nn.InstanceNorm1d(ch_in)

        self.leaky = th.nn.LeakyReLU(0.4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky( self.norm1(out) )

        out = self.conv2(out)
        out = self.leaky( self.norm2(out) )

        out = self.conv3(out)
        out = x + self.leaky( self.norm3(out) )

        return th.relu(out)

def make_encoder(latent_dim):
    return th.nn.Sequential(
        Resnet1D(1, 64, 63),                                                                # (1, 90000) -> (1, 90000)
        th.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=63, stride=2, padding=31),  # (1, 90000) -> (1, 45000) 
        th.nn.ReLU(),
        
        Resnet1D(1, 128, 127),                                                              # (1, 45000) -> (1, 45000)
        th.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=127, stride=2, padding=63), # (1, 45000) -> (1, 22500) 
        th.nn.ReLU(),
        
        Resnet1D(1, 128, 127),                                                              # (1, 22500) -> (1, 25000)
        th.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=127, stride=2, padding=63), # (1, 22500) -> (1, 11250) 
        th.nn.ReLU(),
        
        Resnet1D(1, 256, 255),                                                               # (1, 11250) -> (1, 11250)
        th.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=255, stride=2, padding=127), # (1, 11250) -> (1, 5625) 
        th.nn.ReLU(),
        th.nn.Flatten(1),                                                                    # (1, 5625) -> (5625)
    )

def make_decoder(latent_dim, in_dim):
    return th.nn.Sequential(
        th.nn.Linear(latent_dim, in_dim//16),                                                           # (latent_dim) -> (5625)
        th.nn.ReLU(),
        th.nn.Unflatten(1, (1, 5625)),                                                                 # (5625) -> (1, 5625)

        ResnetTransposed1D(1, 256, 255),
        th.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=256, stride=2, padding=127),  # (1, 5625) -> (1, 11250) 
        th.nn.ReLU(),

        ResnetTransposed1D(1, 128, 127),
        th.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=128, stride=2, padding=63),   # (1, 11250) -> (1, 22500) 
        th.nn.ReLU(),

        ResnetTransposed1D(1, 128, 127),
        th.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=128, stride=2, padding=63),   # (1, 22500) -> (1, 45000) 
        th.nn.ReLU(),

        ResnetTransposed1D(1, 64, 63),
        th.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=64, stride=2, padding=31),   # (1, 45000) -> (1, 90000) 
    )

class VAE(th.nn.Module):
    def __init__(self, latent_dim, in_dim):
        super(VAE, self).__init__()

        self.encoder = make_encoder(latent_dim)

        self.mu = th.nn.Linear(in_dim//16, latent_dim)
        self.log_var = th.nn.Linear(in_dim//16, latent_dim)

        self.decoder = make_decoder(latent_dim, in_dim)

    def forward(self, x):
        x = self.encoder(x)

        mu, log_var = self.mu(x), self.log_var(x)

        z = mu + th.exp(0.5 * log_var) * th.randn_like(mu)

        out = self.decoder(z)

        return out, mu, log_var