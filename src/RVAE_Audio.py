import numpy as np
import torch as th
from torch.utils.data import Dataset

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class Database(Dataset):
    def __init__(self, X):
        self.X = th.from_numpy(X).reshape(-1, 30, 3000)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)

class Encoder(th.nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.fc = th.nn.Linear(in_features=input_dim*num_layers, out_features=input_dim*num_layers//30)
        
        self.input_latent = th.nn.LSTM(input_dim//5, input_dim//15, num_layers//6, batch_first=True)
        self.latent_latent = th.nn.LSTM(input_dim//15, input_dim//15, num_layers//6, batch_first=True)
        
        self.hidden2mu = th.nn.Linear(input_dim*num_layers//45, latent_dim)
        self.hidden2logvar = th.nn.Linear(input_dim*num_layers//45, latent_dim)

        self.input_dim = input_dim

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = x.reshape(x.shape[0], -1, self.input_dim//5)
        
        inverse_idx = th.arange(x.shape[1]-1, -1, -1).to(device) 
        inverse_x   = x.index_select(1, inverse_idx)              # (batch, notes, seconds) -> (batch, notes_reversed, seconds)

        _, (hn_left, _) = self.input_latent(inverse_x)            # (batch, notes_reversed, seconds) -> (notes_reversed, batch, seconds)

        hn_left = hn_left.permute(1, 0, 2)                        # (notes_reversed, batch, seconds) -> (batch, notes_reversed, seconds)
        hn_left = hn_left.index_select(1, inverse_idx)            # (batch, notes_reversed, seconds) -> (batch, notes, seconds)

        _, (hn_right, _) = self.latent_latent(hn_left)            # (batch, notes, seconds) -> (notes, batch, seconds)

        hn_right = hn_right.permute(1, 0, 2)                      # (notes, batch, seconds) -> (batch, notes, seconds)

        hn = th.cat((hn_left, hn_right), dim=2)                   # (batch, notes, seconds) -> (batch, notes, 2*seconds)
        hn = hn.reshape(hn.shape[0], -1)                          # (batch, notes, 2*seconds) -> (batch, 2*seconds*notes)

        mu = self.hidden2mu(hn)
        log_var = self.hidden2logvar(hn)

        return mu, log_var

class Decoder(th.nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.latent_hidden = th.nn.Linear(latent_dim, input_dim*num_layers//45)
        self.decoder_rnn = th.nn.LSTM(input_dim//15, input_dim//5, num_layers//3, batch_first=True)
        self.hidden_generated = th.nn.Linear(input_dim*num_layers//15, input_dim*num_layers)

        self.input_dim = input_dim

    def forward(self, z):
        z = self.latent_hidden(z)                                # (batch, latent_dim) -> (batch, notes*seconds)

        z = z.reshape(z.shape[0], -1, self.input_dim//15)            # (batch, notes*seconds) -> (batch, notes, seconds)

        _, (hn, _) = self.decoder_rnn(z)                         # (batch, notes, seconds) -> (notes, batch, seconds)

        hn = hn.permute(1, 0, 2)                                 # (notes, batch, seconds) -> (batch, notes, seconds)
        hn = hn.reshape(hn.shape[0], -1)                         # (batch, notes, seconds) -> (batch, notes*seconds)

        out = self.hidden_generated(hn)                          # (batch, notes*seconds) -> (batch, notes*seconds)
        out = out.reshape(out.shape[0], -1, self.input_dim)      # (batch, notes*seconds) -> (batch, seconds, notes)

        return out

class VAE(th.nn.Module):
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

        return th.tanh(out), mu, log_var