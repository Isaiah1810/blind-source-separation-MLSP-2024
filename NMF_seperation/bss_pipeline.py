import cupy as cp
import numpy as np
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt

from learn_music_b import train_music_bases
from learn_vocal_b import train_speech_bases
from bss import seperate_audio


# def cosine_sim(v1, v2):
#     return np.einsum('ij,ij->i', v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cosine_sim(v1, v2):
    sims = np.zeros((v1.shape[0]))
    for i in range(v1.shape[0]):
        sims[i] = np.dot(v1[i], v2[i]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[i]))
    return sims

if __name__ == "__main__":

    num_bases = 200
    num_iterations = 200

    data_dir = "../data/train"

    song_path = "../data/train/Actions - One Minute Smile/mixture.wav"

    vocals_path = "../data/train/Actions - One Minute Smile/vocals.wav"


    load_bases = True
    graph_only = False

    if graph_only:
        music, sr = librosa.load("../results/music.wav", duration=30)
        vocals, sr = librosa.load("../results/vocals.wav", duration=30)
    else:
        if (load_bases):
            B_speech = cp.loadtxt("speech_B.txt")
            B_music  = cp.loadtxt("music_B.txt")

        else: 
            B_music = train_music_bases(data_dir, num_bases, num_iterations)
            B_speech = train_speech_bases(data_dir, num_bases, num_iterations)

        music, vocals, sr = seperate_audio(song_path, B_speech, B_music)


    vocal_ref, sr = librosa.load(vocals_path, duration=30)

    song, sr = librosa.load(song_path, duration=30)
    music_ref = song - vocal_ref


    extracted_vocals_mfccs = librosa.feature.mfcc(y=vocals, sr=sr)
    extracted_music_mfccs = librosa.feature.mfcc(y=music, sr=sr)

    ground_vocals_mfccs = librosa.feature.mfcc(y=vocal_ref, sr=sr)
    ground_music_mfccs = librosa.feature.mfcc(y=music_ref, sr=sr)

    cos_sim_music = cosine_sim(extracted_music_mfccs, ground_music_mfccs)
    cos_sim_vocals = cosine_sim(extracted_vocals_mfccs, ground_vocals_mfccs)

    coefs = np.arange(cos_sim_music.shape[0])

    plt.plot(coefs, cos_sim_music, cos_sim_vocals)
    plt.legend(["Music", "Vocals"])
    plt.xlabel("Coefficients")
    plt.ylabel("Cosine Similarity")
    plt.title("Extracted vs Ground MFCC Similarity")
    plt.show()

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

    music_spect = np.abs(librosa.stft(music, n_fft=2048, win_length=1024, hop_length=256))
    vocal_spect = np.abs(librosa.stft(vocals, n_fft=2048, win_length=1024, hop_length=256))

    vocal_ref_spect = np.abs(librosa.stft(vocal_ref, n_fft=2048, win_length=1024, hop_length=256))
    music_ref_spect = np.abs(librosa.stft(music_ref, n_fft=2048, win_length=1024, hop_length=256))

    librosa.display.specshow(music_spect, win_length=1024, hop_length=256)
    plt.title("Isolated Music Spectrogram")
    plt.show()

    librosa.display.specshow(vocal_spect, win_length=1024, hop_length=256)
    plt.title("Isolated Vocals Spectrogram")
    plt.show()

    librosa.display.specshow(music_ref_spect, win_length=1024, hop_length=256)
    plt.title("Music Ground Truth Spectrogram")
    plt.show()

    librosa.display.specshow(vocal_ref_spect, win_length=1024, hop_length=256)
    plt.title("Ground Truth Vocals Spectrogram")
    plt.show()