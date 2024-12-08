import cupy as cp
import librosa
import soundfile as sf
from sklearn.cluster import KMeans  # sklearn KMeans operates on CPU

def kl_divergence(M, B, W):
    M_hat = cp.maximum(B @ W, 1e-10)
    return cp.sum(M * cp.log(M / M_hat + 1e-10))

def nndsvd(M, k, mode="nndsvdar"):
    U, S, Vt = cp.linalg.svd(M, full_matrices=False)
    D, N = M.shape
    W = cp.random.random((k, N)) + 1e-7 if mode == "nndsvdar" else cp.zeros((k, N))

    W[0, :] = cp.sqrt(S[0]) * cp.abs(Vt[0, :])
    for i in range(1, min(k, D)):
        u, v = U[:, i], Vt[i, :]
        u_pos, v_pos = cp.maximum(u, 0), cp.maximum(v, 0)
        norm_u_pos, norm_v_pos = cp.linalg.norm(u_pos), cp.linalg.norm(v_pos)
        W[i, :] = cp.sqrt(S[i] * norm_u_pos * norm_v_pos) * v_pos / norm_v_pos

    if mode == "nndsvda":
        avg = cp.mean(M)
        W[W == 0] = avg
    return W

def nmf(M, B, n_bases=40, thresh=1e-6, n_iterations=200, lambda_sparse=1e-2, lambda_temporal=1e-1, verbose=True, start_weights=None):
    D, N = M.shape
    if start_weights is None:
        W = nndsvd(M, n_bases, mode="nndsvdar")
    else:
        W = start_weights
    print(B.shape, W.shape)
    # Small Change for Numerical Stability   
    epsilon = 1e-10
    i, prev_div = 0, cp.inf

    while i < n_iterations and prev_div > thresh:

        # Only update W, keep B fixed
        W_update = (B.T @ (M / (B @ W + epsilon))) / (B.T @ cp.ones(M.shape) + lambda_sparse + epsilon)

        W *= W_update

        # Ensure Non-Negativity (from numerical errors)
        W = cp.maximum(W, epsilon)

        # KL divergence to monitor progress
        kl = kl_divergence(M, B, W)
        div = cp.abs(kl - prev_div)
        if verbose:
            print(f"Iteration {i}: Change = {div:.2f}, KL Divergence = {kl:.6f}")
        i += 1
        prev_div = kl
    return B, W

def remove_inactive_bases(B, W, threshold=100):
    active = cp.sum(W, axis=1) > threshold
    return B[:, active] + 1e-8, W[active] + 1e-8

def calculate_hnr(signal, sr, fmin=50.0, fmax=500.0):
    harmonic = librosa.effects.harmonic(cp.asnumpy(signal))
    noise = cp.asnumpy(signal) - harmonic
    return 10 * cp.log10(cp.sum(harmonic ** 2) / (cp.sum(noise ** 2) + 1e-10))

def perform_separation(mag, B_speech, B_music, phase, sr=22050, verbose=True):
    n_bases, n_iterations = 400, 200
    lambda_sparse, lambda_temporal = 1, 0.1

    B = cp.concatenate([B_speech, B_music], axis=1)

    # Load weights to update
    # weights = cp.loadtxt("weights.txt")
    # bases = cp.loadtxt("bases.txt")
    bases, weights = nmf(mag, B, n_bases=B.shape[1], n_iterations=n_iterations, 
                         lambda_sparse=lambda_sparse, lambda_temporal=lambda_temporal, verbose=verbose, start_weights=None)

    #cp.savetxt("bases.txt", cp.asnumpy(B))
    #cp.savetxt("weights.txt", cp.asnumpy(weights))

    # Create Reconstructed Speech and Music
    S = B_speech @ weights[:B_speech.shape[1], :]
    M = B_music @ weights[B_speech.shape[1]:, :]

    # Construct Wiener Filter
    wiener_filter = (M ** 2) / (S ** 2 + (M) ** 2 + 1e-10)

    # Apply filter and recreate isolated vocals
    reconstructed_spect = wiener_filter * mag * cp.exp(1j * phase)
    signal_music = librosa.istft(cp.asnumpy(reconstructed_spect), hop_length=256)

    sf.write("music.wav", signal_music, sr)

####################### ANALYSIS FILES ###################################################
    # Create Un_filtered vocals and music for comparison

    signal_vocal = librosa.istft(cp.asnumpy(S * cp.exp(1j * phase)), hop_length=256, n_fft=2048, win_length=1024)
    signal_music = librosa.istft(cp.asnumpy(M * cp.exp(1j * phase)), hop_length=256, n_fft=2048, win_length=1024)
    #harmonic, percussive = librosa.effects.hpss(signal_vocal)
    sf.write("vocals.wav", signal_vocal, sr)
    # sf.write("music_no_mask.wav", signal_music, sr)

    # # Full audio reconstruction for comparison
    # unseparated_recon = librosa.istft(cp.asnumpy((bases @ weights) * cp.exp(1j * phase)), n_fft=2048, win_length=1024)
    # sf.write("full_recon.wav", unseparated_recon ,sr)
    
    return signal_music, signal_vocal

def seperate_audio(path, b_speech, b_music):
    wave, sr = librosa.load(path, sr=None, duration=30)
    spect = cp.asarray(librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256))
    mag = cp.abs(spect)
    phase = cp.angle(spect)
    music, vocals = perform_separation(mag, b_speech, b_music, phase, sr=sr)    
    return music, vocals, sr


if __name__ == "__main__":
    import numpy as np
    b_speech = cp.loadtxt("speech_B.txt")
    b_music  = cp.loadtxt("music_B.txt")
    path = "../data/train/Actions - One Minute Smile/linear_mixture.wav"
    music, vocals, sr = seperate_audio(path, b_speech, b_music)

    # true_vocals, sr = librosa.load("data/train/Actions - One Minute Smile/vocals.wav")

    # import matplotlib.pyplot as plt

    # mfcc_vocals = librosa.feature.mfcc(y=vocals, sr=sr, win_length=int(sr/1000)*32, hop_length=int(sr/1000)*8)
    # mfcc_ground = librosa.feature.mfcc(y=true_vocals, sr=sr, win_length=int(sr/1000)*32, hop_length=int(sr/1000)*8)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(15, 5))

    # # Ground Truth Vocals MFCC
    # plt.subplot(1, 2, 1)
    # librosa.display.specshow(mfcc_ground, x_axis='time', sr=sr)
    # plt.colorbar()
    # plt.title('MFCC - Ground Truth Vocals')
    # plt.xlabel('Time')
    # plt.ylabel('MFCC Coefficients')

    # # Reconstructed Vocals MFCC
    # plt.subplot(1, 2, 2)
    # librosa.display.specshow(mfcc_vocals, x_axis='time', sr=sr)
    # plt.colorbar()
    # plt.title('MFCC - Reconstructed Vocals')
    # plt.xlabel('Time')
    # plt.ylabel('MFCC Coefficients')

    # plt.tight_layout()
    # plt.show()
