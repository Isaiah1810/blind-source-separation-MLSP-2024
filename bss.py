import numpy as np
import librosa 
import scipy
import soundfile as sf


# CREDIT: Kmeans and Kmpp_init functions pulled from Isaiah Weekes' 
# previous 10-301 homework (and slightly modified)

def kl_divergence(M, B, W):
    M_hat = B @ W
   
    M_hat = np.maximum(M_hat, 1e-10)  # Add small constant to avoid log(0)
    
    # Calculate KL Divergence
    kl_div = np.sum(M * np.log(M / M_hat + 1e10)) # For numerical stability
    
    return kl_div

def nndsvd(M, k, variant="nndsvd"):
    """
    NNDSVD algorithm to initialize NMF matrices B and W.
    
    Parameters:
    X: 2D np.array
        The input data matrix to decompose (usually a non-negative matrix).
    k: int
        The number of components (bases) for the decomposition.
    variant: str, optional
        The NNDSVD variant: "nndsvd" for the basic algorithm, "nndsvda" for
        NNDSVD with zeros filled with the average value, "nndsvdar" for
        NNDSVD with zeros filled with small random values.
        
    Returns:
    W: 2D np.array
        The initialized basis matrix (D x K).
    H: 2D np.array
        The initialized weight matrix (K x N).
    """
    D, N = M.shape
    
    # Step 1: Compute the SVD of X
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Step 2: Initialize W and H
    W = np.zeros((D, k))
    H = np.zeros((k, N))
    
    # Step 3: Initialize the first component
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
    
    # Step 4: Initialize the remaining components
    for i in range(1, k):
        u = U[:, i]
        v = Vt[i, :]
        u_pos = np.maximum(u, 0)
        u_neg = np.maximum(-u, 0)
        v_pos = np.maximum(v, 0)
        v_neg = np.maximum(-v, 0)
        
        norm_u_pos = np.linalg.norm(u_pos)
        norm_v_pos = np.linalg.norm(v_pos)
        norm_u_neg = np.linalg.norm(u_neg)
        norm_v_neg = np.linalg.norm(v_neg)
        
        pos_term = norm_u_pos * norm_v_pos
        neg_term = norm_u_neg * norm_v_neg
        
        if pos_term >= neg_term:
            W[:, i] = np.sqrt(S[i] * pos_term) * u_pos / norm_u_pos
            H[i, :] = np.sqrt(S[i] * pos_term) * v_pos / norm_v_pos
        else:
            W[:, i] = np.sqrt(S[i] * neg_term) * u_neg / norm_u_neg
            H[i, :] = np.sqrt(S[i] * neg_term) * v_neg / norm_v_neg
    
    # Step 5: Handle zero elements based on the variant
    if variant == "nndsvda":
        avg = np.mean(M)
        W[W == 0] = avg
        H[H == 0] = avg
    elif variant == "nndsvdar":
        np.random.seed(0)
        W[W == 0] = np.random.random(W[W == 0].shape) * 1e-4
        H[H == 0] = np.random.random(H[H == 0].shape) * 1e-4
    
    return W, H

def nmf(M, n_bases=20, thresh=1e-6, n_iterations=200, lambda_sparse=5e-2, verbose=True):
    D, N = M.shape
    K = n_bases
    B, W = nndsvd(M, n_bases)
    ones = np.ones(M.shape)
    i = 0
    div = np.inf  
    prev = 0
    epsilon = 1e-10 # For numerical stability
    while i < n_iterations and div > thresh:
        B = B * (((M / (B @ W)) @ W.T) / (ones @ W.T + epsilon))
        W = W * ((B.T @ (M / (B @ W))) / (B.T @ ones + lambda_sparse + epsilon))
        kl = kl_divergence(M, B, W)
        div = np.abs(kl - prev)
        if verbose:
            print(f"Iteration {i}: Change = {div:.2f}: Divergance = {kl:.6f}")
        i += 1
        prev = kl
    return B, W


def calculate_hnr(signal, sr, fmin=50.0, fmax=500.0):
    f0, _, _ = librosa.pyin(signal, fmin=fmin, fmax=fmax, sr=sr)
    harmonic_part = librosa.effects.harmonic(signal)
    noise_part = signal - harmonic_part
    hnr = 10 * np.log10(np.sum(harmonic_part ** 2) / np.sum(noise_part ** 2) + 1e-10)
    return hnr


def kmpp_init(X, K):
    """Perform K-Means++ Cluster Initialization.
    
    Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        K: an int where K is the number of cluster centers 
    
    Output:
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
    """
    N = X.shape[0]
    C = [X[np.random.randint(N)]]
    
    for _ in range(K - 1):
        # Compute the distance from each point to the closest center
        dists = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(C), axis=2), axis=1) ** 2
        # Normalize distances to form a probability distribution
        dists /= np.sum(dists)
        # Choose the next center based on the weighted probability distribution
        next_center = X[np.random.choice(N, p=dists)]
        C.append(next_center)
    
    return np.array(C)

def kmeans_loss(X, C, z):
    """Compute the K-means loss."""
    return np.mean(np.sum((X - C[z])**2, axis=1))

def centers_unchanged(c_prev, c_curr):
    return np.allclose(c_prev, c_curr)

def smallest_dist(point, C):
    return np.argmin(np.linalg.norm(C - point, axis=1))

def kmeans(X, K, algo=0):
    """Cluster data X into K clusters using K-means."""
    N = X.shape[0]
    
    # Initialize cluster centers
    if algo == 1:
        C = kmpp_init(X, K)
    else:
        C = X[np.random.choice(N, size=K, replace=False)]
    
    # Initialize labels
    z = np.zeros(N, dtype=int)
    c_prev = np.zeros_like(C)
    
    # Iterate until centers converge
    while not centers_unchanged(c_prev, C):
        c_prev = C.copy()
        
        # Compute cluster assignments using vectorized distance calculation
        z = np.argmin(np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1)
        
        # Update cluster centers by taking the mean of assigned points
        for i in range(K):
            points_in_cluster = X[z == i]
            if len(points_in_cluster) > 0:
                C[i] = np.mean(points_in_cluster, axis=0)
    
    return C, z

if __name__ == "__main__":

    verbose = True

    # Load audio file
    wave, sr = librosa.load("data/train/A Classic Education - NightOwl/mixture.wav")
    
    # Compute the magnitude spectrogram
    spect = librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256)
    mag = np.abs(spect)
    phase = np.angle(spect)
    
    # Perform NMF
    n_bases = 20  
    n_iterations = 200
    bases, weights = nmf(mag, n_bases=n_bases, n_iterations=n_iterations, verbose=True)

    # Perform Kmeans Clustering (with Kmeans++ initialization)
    n_clusters = 5
    if verbose:
        print("Clustering Weights")
    C, z = kmeans(weights, n_clusters, algo=1)


    # Calculate Harmonic-to-noise ratio
    hnr_values = []
    cluster_signals = []
    for cluster_idx in range(n_clusters):
        if verbose:
            print(f"Calculating HNR for cluster {cluster_idx}")
        # Get all data points assigned to this cluster

        cluster_weights = weights[z == cluster_idx]
        cluster_bases = bases[:, z == cluster_idx]

        # If there are no points in the cluster, skip it
        if len(cluster_weights) == 0:
            hnr_values.append(-np.inf)  # Very low HNR for empty clusters
            cluster_signals.append([])
            continue
        
        # Reconstruct audio singal for HNR analysis
        cluster_mag = cluster_bases @ cluster_weights
        cluster_spect = cluster_mag * np.exp(1j * phase)
        cluster_signal = librosa.istft(cluster_spect, n_fft=2048, win_length=1024, hop_length=256)
        # Calculate HNR for the concatenated signal
        hnr_value = calculate_hnr(cluster_signal, sr)
        hnr_values.append(hnr_value)
        cluster_signals.append(cluster_signal)
    
    # Select cluster with highest hnr
    vocal_cluster = np.argmax(hnr_values)

    sf.write("vocals.wav", cluster_signals[vocal_cluster], sr)

    for i in range(len(cluster_signals)):
         sf.write(f"cluster{i}.wav", cluster_signals[i], sr)