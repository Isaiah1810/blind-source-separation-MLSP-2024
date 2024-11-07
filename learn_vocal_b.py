import numpy as np
import librosa
import os

def kl_divergence(M, B, W):
    M_hat = B @ W
    M_hat = np.maximum(M_hat, 1e-10)  # Add small constant to avoid log(0)
    
    # Calculate KL Divergence
    kl_div = np.sum(M * np.log(M / M_hat + 1e-10))  # For numerical stability
    
    return kl_div

def nndsvd(M, k, mode="nndsvd"):
    D, N = M.shape
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    B = np.zeros((D, k))
    W = np.zeros((k, N))
    
    # Initialize the first component
    B[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    W[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
    
    # Initialize the remaining components
    for i in range(1, min(k, D)):
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
            B[:, i] = np.sqrt(S[i] * pos_term) * u_pos / norm_u_pos
            W[i, :] = np.sqrt(S[i] * pos_term) * v_pos / norm_v_pos
        else:
            B[:, i] = np.sqrt(S[i] * neg_term) * u_neg / norm_u_neg
            W[i, :] = np.sqrt(S[i] * neg_term) * v_neg / norm_v_neg
    
    return B, W

def nmf(M, n_bases=40, thresh=1e-6, n_iterations=200, lambda_sparse=1e-1, 
        lambda_temporal=1e-1, verbose=True):
    D, N = M.shape
    K = n_bases
    epsilon = 1e-10  # For numerical stability

    # Use NNDSVD for initialization
    print("initialzing")
    B = np.random.random((D, n_bases))
    W = np.random.random((n_bases, N))
    
    ones = np.ones(M.shape)
    i = 0
    div = np.inf  
    prev = 0
    
    while i < n_iterations and div > thresh:
        # Update B and W
        B = B * (((M / (B @ W)) @ W.T) / (ones @ W.T + epsilon))
        W_update = (B.T @ (M / (B @ W))) / (B.T @ ones + lambda_sparse + epsilon)
        
        # Temporal continuity regularization on W
        for t in range(1, N):
            W_update[:, t] += lambda_temporal * (W[:, t-1] - W[:, t])
        
        W = W * W_update
        B = np.clip(B, epsilon, None)
        W = np.clip(W, epsilon, None)
        
        kl = kl_divergence(M, B, W)
        div = np.abs(kl - prev)
        if verbose:  # Print every iteration
            print(f"Iteration {i}: Change = {div:.2f}: Divergence = {kl:.6f}")
        i += 1
        prev = kl
    
    return B, W

if __name__ == "__main__":
    data_dir = "data/train"
    song_list = os.listdir(data_dir)[:10]
    spect_list = []
    
    for song in song_list:
        print(f"Getting song {song}")
        wave, sr = librosa.load(os.path.join(data_dir, song, 'vocals.wav'))
        spect = np.abs(librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256))
        spect_list.append(spect)
        
    M = np.hstack(spect_list)
    print("Running NMF")
    B, W = nmf(M)

    # Save the basis matrix B to a text file
    np.savetxt("speech_B.txt", B)
