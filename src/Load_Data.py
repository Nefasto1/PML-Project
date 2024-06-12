import numpy as np

import os                                          # To check the folders data
import librosa                                     # To manage the audio files

def load_spectogram():
    # Set the constants
    n_mels = 128
    time_frame = 216
    latent_dim = 16
    batch_size = 10
    epochs = 100
    
    # Load and process audio data
    def load_audio_data(audio_files):
        audio_data = []
    
        for file in audio_files:
            try:
                audio, sampling_rate = librosa.load(file)
    
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels)
                S_dB = librosa.power_to_db(mel_spectrogram)
    
                # If the time_frame is greater than the actual frames, pad with zeros
                if S_dB.shape[1] < time_frame:
                    pad_width = time_frame - S_dB.shape[1]
                    S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
                # If the time_frame is less than the actual frames, truncate the extra frames
                if S_dB.shape[1] > time_frame:
                    S_dB = S_dB[:, :time_frame]
    
                #Normalize the data
                S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
                audio_data.append(S_dB)
    
            except Exception as e:
                print(f"Skipping file {file} due to error: {e}")
    
        return np.array(audio_data)
        
    # Load audio data
    directory = './Music/genres_original/jazz/'
    audio_files = []
        
    #create the audiofiles array
    for file in sorted(os.listdir(directory)):
        audio_files.append(os.path.join(directory, file))
    
    audio_data = load_audio_data(audio_files)
    
    time_frame = audio_data.shape[2]
    
    return audio_data, time_frame
        

def load_audio(genre):
    BASE_PATH = "./Music/genres_original"

    def DatasetLoader(class_):
        music_list = np.array(sorted(os.listdir(BASE_PATH+'/'+class_)))
        TrackSet_1 = [(BASE_PATH)+'/'+class_+'/%s'%(x) for x in music_list]
    
        return TrackSet_1[1:]#, TrackSet_2
    
    def load(file_, duration=30):
        data_, sampling_rate = librosa.load(file_, duration=duration, sr=3000)
            
        data_ = data_.reshape(1, data_.shape[0])
    
        return data_

    start = 10 if genre == "country" else 0
    audio_data = np.array([load(file, 30) for file in DatasetLoader(genre)[start:]])
        
    return audio_data