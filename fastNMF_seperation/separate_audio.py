import numpy as np
import librosa
import librosa
import soundfile as sf
import cupy as cp
from fastnmf_ar import FastNMF


if __name__ == "__main__":

    # Use xp=cp for GPU (my laptop doesn't have enough GPU mem)
    xp = np
    n_iterations = 50
    n_sources = 2
    n_bases = 200
    lambda_sparse = 0
    harmonic = False
    # Load Audio and Get Spectrogram
    print("Loading Audio")
    wave, sr = librosa.load("data/train/Actions - One Minute Smile/linear_mixture.wav", sr=None)
    if harmonic:
        harmonic, percussive = librosa.effects.hpss(wave)
        wave = harmonic
    X = xp.asarray(librosa.stft(wave, n_fft=2048, win_length=2048, center=False, hop_length=256))
    X = xp.expand_dims(X, axis=-1)


    W_speech = np.loadtxt("B_speech.txt")
    W_speech = xp.expand_dims(W_speech, axis=0)
    W_music = np.loadtxt("B_music.txt")
    W_music = xp.expand_dims(W_music, axis=0)
    W = np.vstack([W_speech, W_music])

    # Initial Separator Model
    print("Initailizing Model")
    model = FastNMF(xp=xp, lambda_sparse=lambda_sparse, pretrain=True)
    model.init_nmf(X, X.shape[0], X.shape[1], n_bases, n_sources, X.shape[2], W_pretrain=W)

    # Run Model for n_iterations
    for i in range(n_iterations):
        print(f"Itertation {i+1}")
        model.update_H_only()
    
    # Recover and Save Separated Audio
    print("Recovering Audio")
    rec_spects = model.separate()
    for i in range(n_sources):
        rec_wave = librosa.istft(rec_spects[i, :, :], n_fft=2048, win_length=1024, hop_length=256)
        sf.write(f'Recovered_audio_{i}.wav', rec_wave, sr)

    vocal_spect = W_speech[0] @ model.H[0]
    rec_vocals = librosa.istft(vocal_spect, n_fft=2048, win_length=1024, hop_length=256)
    sf.write("vocals.wav", rec_vocals, sr)