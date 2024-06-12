import numpy as np
import torch as th
from torch.utils.data import Dataset

class Database(Dataset):
    def __init__(self, X):
        self.X = th.from_numpy(X)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)
        
class Encoder(th.nn.Module):
    def __init__(self, n_mels, time_frame, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = th.nn.Linear(in_features=n_mels*time_frame, out_features=1024, bias=True)

        self.mu = th.nn.Linear(in_features=1024, out_features=latent_dim)
        self.log_var = th.nn.Linear(in_features=1024, out_features=latent_dim)

    def forward(self, x):
        out = x.flatten(1)
        out = th.relu(self.fc1(out))
        mu, log_var = self.mu(out), self.log_var(out)

        return mu, log_var

class Decoder(th.nn.Module):
    def __init__(self, latent_dim, n_mels, time_frame):
        super(Decoder, self).__init__()
        self.fc1 = th.nn.Linear(in_features=latent_dim, out_features=512)
        # self.fc2 = th.nn.Linear(in_features=256, out_features=512)
        # self.fc3 = th.nn.Linear(in_features=512, out_features=1024)
        self.out = th.nn.Linear(in_features=512, out_features=n_mels * time_frame)

        self.n_mels = n_mels
        self.time_frame = time_frame

    def forward(self, x):
        out = th.relu(self.fc1(x))
        out = th.sigmoid(self.out(out))
        out = out.reshape(-1, self.n_mels, self.time_frame)

        return out

class VAE(th.nn.Module):
    def __init__(self, n_mels, time_frame, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_mels, time_frame, latent_dim)
        self.decoder = Decoder(latent_dim, n_mels, time_frame)

    def forward(self, x=None, mu=None, log_var=None):
        if not x is None:
            mu, log_var = self.encoder(x)

        if not (mu is None or log_var is None):
            z = mu + th.exp(0.5 * log_var) * th.randn_like(mu)

        out = self.decoder(z)

        return out, mu, log_var