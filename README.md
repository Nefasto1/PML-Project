# Variational Autoencoders for Audio Generation
This repository contains Python notebooks implementing Variational Autoencoders (VAEs) for audio generation and related 
probabilistic machine learning models.

## Project Overview
This project explores the application of VAEs to audio generation tasks. 
We implement and compare several VAE architectures and related probabilistic models, 
focusing on their ability to generate realistic and diverse audio samples.

We worked both with the reconstruction of Mel-Spectogram and and audio data.

## Contents
- [src/VAE_Spectogram.py](https://github.com/Nefasto1/PML-Project/blob/main/src/VAE_Spectogram.py): Simple Variational Auto Encoder to generate Mel-Spectogram
- [src/RVAE_Spectogram.py](https://github.com/Nefasto1/PML-Project/blob/main/src/RVAE_Spectogram.py): Recurrent Variational Auto Encoder to generate Mel-Spectogram
- [src/CNN_Audio.py](https://github.com/Nefasto1/PML-Project/blob/main/src/CNN_Audio.py): Simple CNN-1D Variational Auto Encoder to generate audio data with one channel
- [src/Residual_CNN_Audio.py](https://github.com/Nefasto1/PML-Project/blob/main/src/Residual_CNN_Audio.py): Residual CNN-1D Variational Auto Encoder to generate audio data with one channel
- [src/RVAE_Audio.py](https://github.com/Nefasto1/PML-Project/blob/main/src/RVAE_Audio.py): Recurrent Variational Auto Encoder to generate audio data with one channel
