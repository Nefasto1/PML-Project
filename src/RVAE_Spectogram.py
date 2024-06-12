###################################################################################################
###                                                                                             ###
###   File containing a standard Variational Auto Encoder to be applied to the Mel-Spectograms  ###
###                                                                                             ###
###################################################################################################

import numpy as np
import torch as th
from torch.utils.data import Dataset

class Database(Dataset):
    """
    Database class for the VAE applied to the Mel-Spectograms, just convert the numpy array into torch tensor
    """
    def __init__(self, X):
        self.X = th.from_numpy(X)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)
        
class Encoder(th.nn.Module):
    """
    Encoder class which extend the th.nn.Module class

    Based on the causal structure proposed by Girin et all.
    """
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.input_latent = th.nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)
        self.latent_latent = th.nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)
        self.hidden2mu = th.nn.Linear(input_dim*num_layers*2, latent_dim)
        self.hidden2logvar = th.nn.Linear(input_dim*num_layers*2, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)                                    # (batch, n_mels, time) -> (batch, time, n_mels)
        inverse_x   = x.flip(1)                                   # (batch, time, n_mels) -> (batch, time_reversed, n_mels)

        inverse_z, (hn_left, _) = self.input_latent(inverse_x)            # (batch, time_reversed, n_mels) -> (time_reversed, batch, n_mels)

        z = inverse_z.flip(1)
        
        hn_left = hn_left.permute(1, 0, 2)                        # (time_reversed, batch, n_mels) -> (batch, time_reversed, n_mels)
        hn_left = hn_left.flip(1)            # (batch, time_reversed, n_mels) -> (batch, time, n_mels)

        _, (hn_right, _) = self.latent_latent(z)            # (batch, time, n_mels) -> (time, batch, n_mels)

        hn_right = hn_right.permute(1, 0, 2)                      # (time, batch, n_mels) -> (batch, time, n_mels)

        hn = th.cat((hn_left, hn_right), dim=2)                   # (batch, time, n_mels) -> (batch, time, 2*n_mels)
        hn = hn.reshape(hn.shape[0], -1)                          # (batch, time, 2*n_mels) -> (batch, 2*n_mels*time)

        mu = self.hidden2mu(hn)
        log_var = self.hidden2logvar(hn)

        return mu, log_var

class Decoder(th.nn.Module):
    """
    Decoder class which extend the th.nn.Module class

    Based on the causal structure proposed by Girin et all.
    """
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.latent_hidden = th.nn.Linear(latent_dim, input_dim*num_layers)
        self.decoder_rnn = th.nn.LSTM(input_dim, input_dim//4, num_layers, batch_first=True)
        self.hidden_generated = th.nn.Linear(input_dim*num_layers//4, input_dim*num_layers)

        self.input_dim = input_dim

    def forward(self, z):
        z = self.latent_hidden(z)                                # (batch, latent_dim) -> (batch, n_mels*time)

        z = z.reshape(z.shape[0], -1, self.input_dim)            # (batch, n_mels*time) -> (batch, time, n_mels)

        _, (hn, _) = self.decoder_rnn(z)                         # (batch, time, n_mels) -> (time, batch, n_mels)

        hn = hn.permute(1, 0, 2)                                 # (time, batch, n_mels) -> (batch, time, n_mels)
        hn = hn.reshape(hn.shape[0], -1)                         # (batch, time, n_mels) -> (batch, time*n_mels)

        out = self.hidden_generated(hn)                          # (batch, time*n_mels) -> (batch, time*n_mels)
        out = out.reshape(out.shape[0], self.input_dim, -1)      # (batch, time*n_mels) -> (batch, n_mels, time)

        return out

class VAE(th.nn.Module):
    """
    VAE class which extend the th.nn.Module class

    Based on the causal structure proposed by Girin et all.
    """
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, num_layers)
        self.decoder = Decoder(input_dim, latent_dim, num_layers)

    def forward(self, x=None, mu=None, log_var=None):
        if not x is None:
            mu, log_var = self.encoder(x)

        if not (mu is None or log_var is None):
            z = mu + th.exp(0.5 * log_var) * th.randn_like(mu)

        out = self.decoder(z)

        return out, mu, log_var