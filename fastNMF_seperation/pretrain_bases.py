import numpy as np
import librosa
import os
import soundfile as sf
import cupy as cp
from fastnmf_ar import FastNMF


if __name__ == "__main__":
    data_dir = "data/train"
    song_list = os.listdir(data_dir)[0:20]
    spect_list = []
    
    for song in song_list:
        print(f"Loading song {song}")
        wave, sr = librosa.load(os.path.join(data_dir, song, 'vocals.wav'), duration=30)
        spect = np.asarray(librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256))
        spect_list.append(spect)

    # for song in song_list:
    #     print(f"Getting song {song}")
    #     wave1, sr = librosa.load(os.path.join(data_dir, song, 'mixture.wav'), duration=30)
    #     wave2, sr = librosa.load(os.path.join(data_dir, song, 'vocals.wav'), duration=30)
    #     wave = wave1 - wave2
    #     spect = librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256)
    #     spect_list.append(spect)
    
    W = np.loadtxt("bases_k_4.txt")
    W = np.expand_dims(W, 0)

    M = np.hstack(spect_list)
    M = np.expand_dims(M, axis=-1)
    print("Initializing Model")
    model = FastNMF(xp=np, pretrain=True)
    model.init_nmf(M, M.shape[0], M.shape[1], 4, 1, 1, W_pretrain=W)

    print("Running NMF")
    for i in range(100):
        print(f"Iteration {i+1}")
        model.update()

    np.savetxt("bases_k_4.txt", model.W[0])

    