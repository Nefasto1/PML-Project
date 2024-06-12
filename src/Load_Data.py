###########################################################################
###                                                                     ###
###   File containing some functions to load and pre-process the data   ###
###                                                                     ###
###########################################################################

import numpy as np
import os                                          # To check the folders data
import librosa                                     # To manage the audio files

def load_spectogram():
    """
    Function to load the audio and convert them into spectogram

    Output
    ------
       audio_data: np.array[float, float, float]
          "tensor" containing the Mel-Spectogram of size (#songs, n_mels, time_frame)
       time_frame: int
          number of time_frames
    """

    # Define the constants
    n_mels = 128
    time_frame = 216
    
    # Set the audio directory
    directory = './Music/genres_original/jazz/'
    audio_files = []
        
    # Create the audio file array ordered by name
    for file in sorted(os.listdir(directory)):
        audio_files.append(os.path.join(directory, file))

    
    audio_data = []

    for file in audio_files:
        try:
            # Load the audio data
            audio, sampling_rate = librosa.load(file)

            # Convert the audio into the mel_spectogram and convert the power scale to the decibels scale
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels)
            S_dB = librosa.power_to_db(mel_spectrogram)

            # If the time_frame is greater than the actual frames, pad with zeros
            if S_dB.shape[1] < time_frame:
                pad_width = time_frame - S_dB.shape[1]
                S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')

            # If the time_frame is less than the actual frames, truncate the extra frames
            if S_dB.shape[1] > time_frame:
                S_dB = S_dB[:, :time_frame]

            # Normalize the data
            S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
            audio_data.append(S_dB)

        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")
    
    time_frame = audio_data.shape[2]
    
    return audio_data, time_frame

def load_audio(genre):
    """
    Function to load the audio

    Output
    ------
       audio_data: np.array[float, float, float]
          "tensor" containing the Mel-Spectogram of size (#songs, channels=1, notes=30*3000)   
    """

    def DatasetLoader(class_):
        """
        Function to get the audio paths

        Parameters
        ----------
        class_: str
            genre of the audio

        Output
        ------
        paths: list[str]
            list containing the paths of the audios
        """
        
        music_list = np.array(sorted(os.listdir(BASE_PATH+'/'+class_)))
        paths = ['./Music/genres_original/' + class_ + f'/{x}' for x in music_list]
    
        return paths[1:]
    
    def load(file, duration=30):
        """
        Function to load the data and reshape them into (#channels, #notes)

        Parameters
        ----------
        file: str
            Path of the file to load
        duration: int
            The audio length in seconds

        Output
        ------
        data: np.array[float, float]
            Matrix containing the 3000 notes for each second
        """
        data, sampling_rate = librosa.load(file, duration=duration, sr=3000)
        data = data.reshape(1, data.shape[0])
    
        return data

    start = 10 if genre == "country" else 0
    audio_data = np.array([load(file, 30) for file in DatasetLoader(genre)[start:]])
        
    return audio_data