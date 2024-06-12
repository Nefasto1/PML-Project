import numpy as np
import torch as th
from tqdm import trange                            # To display the loading bar for the train
import os
import matplotlib.pyplot as plt                    # To display the results



def save_checkpoint(model, loss, i):
    # save_model(model, f"./Models/model_{i}.safetensors")

    if i != 1: #or "losses.npy" in os.listdir("./Models"):
        with open("./Models/losses.npy", "rb") as f:
            losses = np.load(f)
    else:
        losses = np.array([])

    losses = np.append(losses, loss)

    with open("./Models/losses.npy", "wb") as f:
        np.save(f, losses)

def load_checkpoint(model, idx=None):
    idx = 1
    if idx is not None:
        try:
            _ = load_model(model, f"./Models/model_{idx}.safetensors")
            print(f"Checkpoint {idx} found and loaded")
            return model, idx
        except:
            print(f"Checkpoint {idx} not found, loading the latest checkpoint")

    files = os.listdir("./Models")
    files = [model for model in files if "model" in model]
    if files:
        idxs = [ int( model.split(".")[0].split("_")[1] ) for model in files]
        idxs.sort()
        idx = idxs[-1]

        print(f"Loaded checkpoint {idx}")
        _ = load_model(model, f"./Models/model_{idx}.safetensors")
        return model, idx

    print("No checkpoint found, returning input model")
    return model, 0

def plot_losses(start=0, stop=-1):
    if not "losses.npy" in os.listdir("./Models"):
        print("No checkpoint found, train a model first")

    with open("./Models/losses.npy", "rb") as f:
        losses = np.load(f)

    plt.plot(range(len(losses[start:stop])), losses[start:stop])

def train(model, train_loader, optimizer, criterion, device, idx, epochs=101):
    bar = trange(1, epochs, desc="Loss: ?")
    for epoch in bar:
        cum_loss = 0
        for batch in train_loader: #tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device=device)
            out, mu, log_var = model(batch.float())
    
            loss = criterion(out, mu, log_var, batch.float())
            cum_loss += loss.item()
    
            loss.backward()
            optimizer.step()
    
        cum_loss = cum_loss/len(train_loader)
        bar.set_description(f"Loss: {cum_loss}")
    
        save_checkpoint(model, [cum_loss], epoch+idx)

    return model