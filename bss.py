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

def nmf(M, B, n_bases=80, thresh=1e-6, n_iterations=200, lambda_sparse=1e-2, lambda_temporal=1e-1, verbose=True):
    D, N = M.shape
    W = nndsvd(M, n_bases, mode="nndsvdar")
    epsilon = 1e-10
    i, prev_div = 0, cp.inf

    while i < n_iterations and prev_div > thresh:
        # Only update W, keep B fixed
        W_update = (B.T @ (M / (B @ W + epsilon))) / (B.T @ cp.ones(M.shape) + lambda_sparse + epsilon)
        for t in range(1, N):
            W_update[:, t] += lambda_temporal * (W[:, t-1] - W[:, t])
        W *= W_update
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

def perform_separation(mag, B, n_clusters=2, sr=22050, verbose=True):
    n_bases, n_iterations = 80, 200
    lambda_sparse, lambda_temporal = 0.1, 0.1

    # Keep B fixed during NMF
    bases, weights = nmf(mag, B, n_bases=n_bases, n_iterations=n_iterations, 
                         lambda_sparse=lambda_sparse, lambda_temporal=lambda_temporal, verbose=verbose)

    cp.savetxt("un_clustered_bases.txt", cp.asnumpy(bases))
    cp.savetxt("weights.txt", cp.asnumpy(weights))

    # Remove inactive bases
    bases, weights = remove_inactive_bases(bases, weights, threshold=cp.max(weights) * 0.01)

    # Perform K-means clustering (needs data to be moved to CPU)
    weights_cpu = cp.asnumpy(weights.T)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
    labels = kmeans.fit_predict(weights_cpu)

    hnr_values, cluster_signals = [], []
    for cluster_idx in range(n_clusters):
        cluster_weights = weights[:, labels == cluster_idx]
        cluster_bases = bases[:, labels == cluster_idx]
        S = cluster_bases @ cluster_weights

        wiener_filter = (S ** 2) / (S ** 2 + (mag - S) ** 2 + 1e-10)
        reconstructed_spect = wiener_filter * mag * cp.exp(1j * cp.angle(cp.asarray(librosa.stft(cp.asnumpy(wave), n_fft=2048, win_length=1024, hop_length=256))))
        cluster_signal = librosa.istft(cp.asnumpy(reconstructed_spect), hop_length=256)
        
        hnr_values.append(calculate_hnr(cluster_signal, sr))
        cluster_signals.append(cluster_signal)

    vocal_cluster = cp.argmax(hnr_values)
    sf.write("vocals.wav", cluster_signals[vocal_cluster], sr)

    for i, signal in enumerate(cluster_signals):
        sf.write(f"cluster{i}.wav", signal, sr)

if __name__ == "__main__":
    wave, sr = librosa.load("data/train/Actions - One Minute Smile/linear_mixture.wav", sr=None)
    mag = cp.abs(cp.asarray(librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256)))
    
    b_speech = cp.loadtxt("speech_B.txt")
    b_music = cp.loadtxt("music_B.txt")
    B = cp.concatenate([b_speech, b_music], axis=1)
    
    perform_separation(mag, B, sr=sr)
