import cupy as cp
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt

from learn_music_b import train_music_bases
from learn_vocal_b import train_speech_bases
from bss import seperate_audio
if __name__ == "__main__":

    num_bases = 200
    num_iterations = 200

    data_dir = "../data/train"

    song_path = "../data/train/Actions - One Minute Smile/linear_mixture.wav"

    vocals_path = "../data/train/Actions - One Minute Smile/vocals.wav"


    load_bases = True

    if (load_bases):
        B_speech = cp.loadtxt("speech_B.txt")
        B_music  = cp.loadtxt("music_B.txt")

    else: 
        B_music = train_music_bases(data_dir, num_bases, num_iterations)
        B_speech = train_speech_bases(data_dir, num_bases, num_iterations)

    music, vocals, sr = seperate_audio(song_path, B_speech, B_music)

    extracted_vocals_mfccs = librosa.feature.mfcc(vocals, sr)
    extracted_music_mfccs = librosa.feature.mfcc(music, sr)

    librosa.display.specshow(extracted_vocals_mfccs, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.title('Extracted Vocals MFCCs')
    plt.show()

    librosa.display.specshow(extracted_vocals_mfccs, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.title('Extracted Music MFCCs')
    plt.show()

    vocal_ref, sr = librosa.load(vocals_path)

    song, sr = librosa.load(song_path)
    music_ref = song - vocal_ref

    ground_vocals_mfccs = librosa.feature.mfcc(vocal_ref, sr)
    ground_music_mfccs = librosa.feature.mfcc(music_ref, sr)

    librosa.display.specshow(ground_vocals_mfccs, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.title('Ground Truth Vocals MFCCs')
    plt.show()

    librosa.display.specshow(ground_music_mfccs, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.title('Ground Truth Music MFCCs')
    plt.show()

    