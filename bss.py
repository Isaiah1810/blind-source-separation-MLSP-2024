import numpy as np
import librosa 
import scipy
import soundfile as sf



def nmf(M, n_bases=5, n_iterations=500, lambda_sparse=0.05):
    # Dimensionality Constants
    D = M.shape[0] 
    N = M.shape[1]
    K = n_bases
    # Initialize B and W randomly
    B = np.random.rand(D, K)
    W = np.random.rand(K, N)
    ones = np.ones(M.shape)
    for i in range(n_iterations):
        B = B * (((M / (B @ W)) @ W.T) / (ones @ W.T))
        W = W * ((B.T @ (M / (B @ W)))/ (B.T @ ones))
        print(i)
    return B, W



if __name__ == "__main__":
    wave, sr = librosa.load("data/train/A Classic Education - NightOwl/mixture.wav")
    spect = librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256)
    mag = np.abs(spect)
    phase = np.angle(spect)
    bases, weights = nmf(mag)
    for i in range(bases.shape[1]):
        comp = (bases[:, i].reshape((mag.shape[0], 1)) @ weights[i, :].reshape((1, mag.shape[1]))) * np.exp(1j * phase)
        wave_base = librosa.istft(comp, n_fft=2048, win_length=1024, hop_length=256)
        sf.write(f"results/base_{i}.wav", wave_base, sr)
