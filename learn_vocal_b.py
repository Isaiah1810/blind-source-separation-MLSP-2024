import cupy as cp
import librosa
import os

def kl_divergence(M, B, W):
    M_hat = B @ W
    M_hat = cp.maximum(M_hat, 1e-10)  # Add small constant to avoid log(0)
    
    # Calculate KL Divergence
    kl_div = cp.sum(M * cp.log(M / M_hat + 1e-10))  # For numerical stability
    
    return kl_div

def nndsvd(M, k):
    D, N = M.shape
    U, S, Vt = cp.linalg.svd(M, full_matrices=False)
    
    B = cp.zeros((D, k))
    W = cp.zeros((k, N))
    
    # Initialize the first component
    B[:, 0] = cp.sqrt(S[0]) * cp.abs(U[:, 0])
    W[0, :] = cp.sqrt(S[0]) * cp.abs(Vt[0, :])
    
    # Initialize the remaining components
    for i in range(1, k):
        u = U[:, i]
        v = Vt[i, :]
        u_pos, u_neg = cp.maximum(u, 0), cp.maximum(-u, 0)
        v_pos, v_neg = cp.maximum(v, 0), cp.maximum(-v, 0)
        
        norm_u_pos = cp.linalg.norm(u_pos) + 1e-10
        norm_v_pos = cp.linalg.norm(v_pos) + 1e-10
        norm_u_neg = cp.linalg.norm(u_neg) + 1e-10
        norm_v_neg = cp.linalg.norm(v_neg) + 1e-10
        
        pos_term = norm_u_pos * norm_v_pos
        neg_term = norm_u_neg * norm_v_neg
        
        if pos_term >= neg_term:
            B[:, i] = cp.sqrt(S[i] * pos_term) * u_pos / norm_u_pos
            W[i, :] = cp.sqrt(S[i] * pos_term) * v_pos / norm_v_pos
        else:
            B[:, i] = cp.sqrt(S[i] * neg_term) * u_neg / norm_u_neg
            W[i, :] = cp.sqrt(S[i] * neg_term) * v_neg / norm_v_neg
    
    return B, W

def nmf(M, n_bases=40, thresh=1e-6, n_iterations=200, lambda_sparse=1e-1, 
        lambda_temporal=1e-1, verbose=True):
    D, N = M.shape
    epsilon = 1e-10  # For numerical stability

    # Use NNDSVD for initialization
    B, W = nndsvd(M, n_bases)
    
    ones = cp.ones(M.shape)
    i, prev_div = 0, cp.inf
    
    while i < n_iterations and prev_div > thresh:
        # Update B and W
        B_update = ((M / (B @ W + epsilon)) @ W.T) / (ones @ W.T + epsilon)
        B *= B_update
        B = cp.maximum(B, epsilon)  # Ensure non-negativity

        W_update = (B.T @ (M / (B @ W + epsilon))) / (B.T @ ones + lambda_sparse + epsilon)
        
        # Temporal continuity regularization on W
        for t in range(1, N):
            W_update[:, t] += lambda_temporal * (W[:, t-1] - W[:, t])
        
        W *= W_update
        W = cp.maximum(W, epsilon)  # Ensure non-negativity

        # Calculate divergence for convergence check
        kl = kl_divergence(M, B, W)
        prev_div = cp.abs(kl - prev_div)

        if verbose:
            print(f"Iteration {i}: Change = {prev_div:.2f}, KL Divergence = {kl:.6f}")
        
        i += 1
        prev_div = kl
    
    return B, W

if __name__ == "__main__":
    data_dir = "data/train"
    song_list = os.listdir(data_dir)[:50]
    spect_list = []
    
    for song in song_list:
        print(f"Loading song {song}")
        wave, sr = librosa.load(os.path.join(data_dir, song, 'vocals.wav'), duration=30)
        spect = cp.array(cp.abs(cp.array(librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256))))
        spect_list.append(spect)
        
    M = cp.hstack(spect_list)
    print("Running NMF")
    B, W = nmf(M)

    # Transfer the result to CPU and save the basis matrix B to a text file
    cp.savetxt("speech_B.txt", cp.asnumpy(B))
