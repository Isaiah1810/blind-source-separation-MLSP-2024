import numpy as np
import cupy as cp
import librosa
'''
AR-FastNMF Algorithm From This Paper:

Kouhei Sekiguchi, Yoshiaki Bando, Aditya Arie Nugraha, Mathieu Fontaine, 
Kazuyoshi Yoshii: Autoregressive Fast Multichannel Nonnegative Matrix
Factorization for Joint Blind Source Separation and Dereverberation, 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
2021.
'''

epsilon = 5e-2

class FastNMF():
    def __init__(self, xp=np, ar_taps=3, norm_t=10):
        self.xp = xp
        self.ar_taps = ar_taps
        self.iter = 0
        self.norm_t = norm_t

    def normalize(self):
        phi_F = self.xp.einsum("fij, fij -> f", self.Q, self.Q.conj()).real / self.M
        self.P /= self.xp.sqrt(phi_F)[:, None, None]
        self.W /= phi_F[None, :, None]

        mu_N = self.G.sum(axis=1)
        self.G /= mu_N[:, None]
        self.W *= mu_N[:, None, None]

        nu_NK = self.W.sum(axis=1)
        self.W /= nu_NK[:, None]
        self.H *= nu_NK[:, :, None]


        self.Px = self.xp.einsum("fmi, fti -> ftm", self.P, self.Xbar)
        self.Px_power = self.xp.abs(self.Px) ** 2

        # Compute Y and Power Spectral Density
        self.PSD = self.W @ self.H + epsilon
        self.Y = self.xp.einsum("nft, nm -> ftm", self.PSD, self.G) + epsilon

    def init_nmf(self, X, F, T, K, N, M, init='Uniform'):
        '''
        X: Observation Spectrogram
        F: Number of Frequencies
        T: Time
        K: Number of Bases
        N: Number of Sources
        init: 
            Uniform - Uniform distribution
            Gaussian - Mean Shifted Gaussian Distribution
            NNSVD - Non-negative Double Singular Value Decomposition
        '''
        self.X = X
        self.F = F
        self.T = T
        self.K = K
        self.N = N
        self.M = M

        self.Xbar = self.xp.zeros([F, T, M * (self.ar_taps + 1)], dtype=complex)
        self.Xbar[:, :, : M] = X
        for i in range(self.ar_taps):
           self.Xbar[:, self.ar_taps + i :, (i + 1) * M : (i + 2) * M] = X[:, : -(self.ar_taps + i)]


        # Initialize W, H, G, Q
        if init=='Gaussian':
            pass
        elif init=="NNDSVD":
            pass
        elif init=='Uniform':
            self.W = self.xp.random.random((N, F, K))
            self.H = self.xp.random.random((N, K, T))
            self.Q = self.xp.tile(self.xp.eye(M), (F, 1, 1)).astype(complex)
            self.P = self.xp.zeros([F, M, M * (self.ar_taps + 1)])
            self.P[:, :, : M] = self.Q
            self.G = self.xp.ones([N, M], dtype=float) * epsilon
            for m in range(M):
                self.G[m % N, m] = 1
            self.G /= self.G.sum(axis=1)[:, None]  

            self.normalize()


        else:
            raise ValueError("Invalid Initialization Parameter")
        
    def update(self):
        # Multiplicative Update for Weights W
        num_mul = self.xp.einsum("nm, ftm -> nft", self.G, self.Px_power / (self.Y ** 2))
        denom_mul = self.xp.einsum("nm, ftm -> nft", self.G, 1 / self.Y)
        numerator = self.xp.einsum("nkt, nft -> nfk", self.H, num_mul)
        denominator = self.xp.einsum("nkt, nft -> nfk", self.H, denom_mul)
        self.W *= self.xp.sqrt(numerator / denominator)

        # Update Y and Power Spectral Density
        self.PSD = self.W @ self.H + epsilon
        self.Y = self.xp.einsum("nft, nm -> ftm", self.PSD, self.G) + epsilon

        # Multiplicative Update for Activation Matrix H
        num_mul = self.xp.einsum("nm, ftm -> nft", self.G, self.Px_power/ (self.Y ** 2))
        denom_mul = self.xp.einsum("nm, ftm -> nft", self.G, 1 / self.Y)
        numerator = self.xp.einsum("nfk, nft -> nkt", self.W, num_mul)
        denominator = self.xp.einsum("nfk, nft -> nkt", self.W, denom_mul)
        self.H *= self.xp.sqrt(numerator / denominator)

        # Update Y and Power Spectral Density
        self.PSD = self.W @ self.H + epsilon
        self.Y = self.xp.einsum("nft, nm -> ftm", self.PSD, self.G) + epsilon

        # Update Spacial Covariance Matrix G 
        numerator = self.xp.einsum("nft, ftm -> nm", self.PSD, self.Px_power / (self.Y ** 2))
        denominator = self.xp.einsum("nft, ftm -> nm", self.PSD, 1 / self.Y)
        self.G *= self.xp.sqrt(numerator / denominator)

        # Update Y
        self.Y = self.xp.einsum("nft, nm -> ftm", self.PSD, self.G) + epsilon

        # Update Diagonalization Parameter Q with Iterative Projection
        for m in range(self.M):
            Vinv = self.xp.linalg.inv(self.xp.einsum("fti, ftj, ft -> fij", self.Xbar, 
                                           self.Xbar.conj(), 1 / self.Y[..., m]) 
                                           / self.T)
            u = self.xp.linalg.inv(self.P[:, :, : self.M])[:, :, m]
            denominator = self.xp.sqrt(
                self.xp.einsum("fi, fij, fj -> f", u.conj(), Vinv[:, : self.M, : self.M], u)
            )
            self.P[:, m] = self.xp.einsum(
                "fi, f -> fi",
                self.xp.einsum("fij, fj -> fi", Vinv[..., : self.M], u),
                1 / denominator).conj()

        self.Q = self.P[:, :, : self.M]

        if self.iter % self.norm_t == 0:
            self.normalize()
        else:
            self.Px = self.xp.einsum("fmi, fti -> ftm", self.P, self.Xbar)
            self.Px_power = self.xp.abs(self.Px) ** 2

    def separate(self, index=0):
        Y_multi = self.xp.einsum("nft, nm -> nftm", self.PSD, self.G)
        self.Y = Y_multi.sum(axis=0)
        self.Px = self.xp.einsum("fmi, fti -> ftm", self.P, self.Xbar)
        Qinv = self.xp.linalg.inv(self.Q)

        self.separated_spec = self.xp.einsum(
            "fj, ftj, nftj -> nft", Qinv[:, index], self.Px / self.Y, Y_multi
        )
        return self.separated_spec

if __name__ == "__main__":
    import librosa
    import soundfile as sf
    import cupy as cp

    # Use xp=cp for GPU (my laptop doesn't have enough GPU mem)
    model = FastNMF(xp=np)

    print("Loading Audio")
    wave, sr = librosa.load("data/train/Actions - One Minute Smile/linear_mixture.wav", sr=None)
    X = librosa.stft(wave, n_fft=2048, win_length=1024, hop_length=256)
    X = np.expand_dims(X, axis=-1)
    
    print("Initailizing Model")
    model.init_nmf(X, X.shape[0], X.shape[1], 4, 4, X.shape[2])


    n_iterations = 50
    for i in range(n_iterations):
        print(f"Itertation {i+1}")
        model.update()
    print("Recovering Audio")
    rec_spect = model.separate()
    rec_wave = librosa.istft(rec_spect, n_fft=2048, win_length=1024, hop_length=256)
    sf.write('Recovered_audio.wav', rec_wave, sr)