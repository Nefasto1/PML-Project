#################################################################
###                                                           ###
###   File Containing some utility functions for the models   ###
###                                                           ###
#################################################################

import numpy as np
import torch as th
from tqdm import trange                            # To display the loading bar for the train
import os
import matplotlib.pyplot as plt                    # To display the results

def save_checkpoint(model, loss, i):
    """
    Function to save the model and loss checkpoint

    Parameters:
    model: th.nn.Module
        Model to save
    loss: np.array[float]
        Array containing the new losses to add to the previous chekpoint
    i: int
        Integer equal to the number of checkpoint (name of the model to save)
    """
    # Save the model into the storage
    # save_model(model, f"./Models/model_{i}.safetensors")

    # If isn't the first iteration, add the losses to the previous checkpoint, otherwise initialize the array
    if i != 1: #or "losses.npy" in os.listdir("./Models"):
        with open("./Models/losses.npy", "rb") as f:
            losses = np.load(f)
    else:
        losses = np.array([])

    # Append the losses to the previous array
    losses = np.append(losses, loss)

    # Save the array
    with open("./Models/losses.npy", "wb") as f:
        np.save(f, losses)

def load_checkpoint(model, idx=None):
    """
    Function to load a model checkpoint

    Parameters
    ----------
    model: th.nn.Module
        An istance of the model of the same class of the requested checkpoint which will be initializated
    idx: int (optional)
        An index relative to the checkpoint to load, if missing or incorrect load the latest
        
    Output
    ------
    model: th.nn.Module
        The loaded model
    idx: 
        The index of the loaded model
    """
    # If requested a specific idx, try to load it. If error load the latest
    if idx is not None:
        try:
            _ = load_model(model, f"./Models/model_{idx}.safetensors")
            print(f"Checkpoint {idx} found and loaded")
            return model, idx
        except:
            print(f"Checkpoint {idx} not found, loading the latest checkpoint")
            
    # Check the available checkpoints
    files = os.listdir("./Models")
    files = [model for model in files if "model" in model]
    
    # If there are checkpoints, load the latest
    if files:
        idxs = [ int( model.split(".")[0].split("_")[1] ) for model in files]
        idxs.sort()
        idx = idxs[-1]

        print(f"Loaded checkpoint {idx}")
        _ = load_model(model, f"./Models/model_{idx}.safetensors")
        return model, idx

    # If there are no checkpoints available return the initial model
    print("No checkpoint found, returning input model")
    return model, 0

def plot_losses(start=0, stop=1e16):
    """
    Function to plot the saved losses

    Parameters
    ----------
    start: int (optional)
        Starting index of the plot
    stop: int (optional)
        Ending index of the plot
    """
    # If no checkpoints, stop the pocess
    if not "losses.npy" in os.listdir("./Models"):
        print("No checkpoint found, train a model first")
        return

    # If there are some checpoints, plot the losses
    with open("./Models/losses.npy", "rb") as f:
        losses = np.load(f)

    plt.plot(range(start, min(stop, len(losses[start:stop]))), losses[start:stop])    
    plt.show()

def train(model, train_loader, optimizer, criterion, device, idx, epochs=100):
    """
    Standard function to train the given model

    Parameters
    ----------
    model: th.nn.Module
        Model to train
    train_loader: th.utils.data.DataLoader
        Train data
    optimizer: th.optim.Optimizer
        Optimizer to use
    criterion: function
        Function that evaluate the loss given y_hat and y
    device: th.device
        Device on which do the train
    idx: int
        Index of the last checkpoint
    epochs: int
        Number of epochs for the train

    Output
    ------
    model: th.nn.Module
        The trained model
    """

    # Train the model for the number of epochs
    bar = trange(1, epochs+1, desc="Loss: ?")
    for epoch in bar:
        # For each batch update the weights
        cum_loss = 0
        for batch in train_loader:
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