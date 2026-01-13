import numpy as np

# ---------------- PCA per window ----------------
def pca_plane(window):
    """
    window: (3,N) or (N,3) raw accel
    returns pc1, pc2 (length N), and eigenvalues
    """
    w = np.asarray(window, float)
    X = w.T if w.shape[0] == 3 else w  # (N,3)
    Xc = X - X.mean(axis=0, keepdims=True)  # PCA centers

    C = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pc = Xc @ eigvecs
    return pc[:, 0], pc[:, 1], eigvals

# ---------------- Repo ButterworthLowPassFilter (fs=30) ----------------
class ButterworthLowPass30Hz:
    # LowPassOrder=5 coefficients from Trajectory.cs
    b = np.array([0.008114715950794,
                  0.032458863803176,
                  0.048688295704763,
                  0.032458863803176,
                  0.008114715950794], dtype=float)

    a = np.array([1.0,
                 -2.101775724168813,
                  1.915053121664871,
                 -0.823185547634419,
                  0.139743605351063], dtype=float)

    def __init__(self):
        self.N = 5
        self.inp = None
        self.out = None
        self.pos = 0

    def filter_sample(self, x):
        if self.inp is None:
            self.inp = np.full(self.N, x, dtype=float)
            self.out = np.full(self.N, x, dtype=float)
            self.pos = -1
            return float(x)

        self.pos = (self.pos + 1) % self.N
        self.inp[self.pos] = x
        self.out[self.pos] = 0.0

        j = self.pos
        acc = 0.0
        for i in range(self.N):
            acc += self.b[i] * self.inp[j] - self.a[i] * self.out[j]
            j = (j - 1) % self.N

        self.out[self.pos] = acc
        return float(acc)

    def filter_series(self, xs):
        xs = np.asarray(xs, float)
        self.inp = None
        self.out = None
        self.pos = 0
        return np.array([self.filter_sample(v) for v in xs], dtype=float)

def lowpass_3p5_repo(x, z, fs=30):
    if fs != 30:
        raise ValueError("This implementation matches the repo's 30Hz coefficients.")
    xf = ButterworthLowPass30Hz().filter_series(x)
    zf = ButterworthLowPass30Hz().filter_series(z)
    return xf, zf

# ---------------- Stats helpers ----------------
def med(x): return float(np.median(np.asarray(x, float)))

def mad_like_repo(x):
    x = np.asarray(x, float)
    m = np.median(x)
    return float(np.median(np.abs(x - m)))

# ---------------- Ellipse like Ellipse.cs ----------------
def ellipse_ab_area_perim(x, z):
    x = np.asarray(x, float); z = np.asarray(z, float)
    cx = float(np.mean(x)); cz = float(np.mean(z))
    dx = x - cx; dz = z - cz
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    sumXX = float(np.sum(dx*dx) / (n - 1))
    sumXZ = float(np.sum(dx*dz) / (n - 1))
    sumZZ = float(np.sum(dz*dz) / (n - 1))
    cov = np.array([[sumXX, sumXZ],[sumXZ, sumZZ]], dtype=float)

    eigvals = np.linalg.eigvalsh(cov)  # ascending
    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))

    chisquare_val = 2.4477
    a = chisquare_val * np.sqrt(max(lam_max, 0.0))
    b = chisquare_val * np.sqrt(max(lam_min, 0.0))

    area = float(np.pi * a * b)
    perim = float((np.pi/2.0) * np.sqrt(2*a*a + 2*b*b))
    return float(a), float(b), area, perim

# ---------------- FrequencyStats like FrequencyStats.cs ----------------
def hamming_repo(N):
    if N <= 1:
        raise ValueError("signalLength must be > 1")
    n = N - 1
    i = np.arange(N, dtype=float)
    return 0.54 - 0.46 * np.cos(2*np.pi * (i / n))

def pxx_fx_repo(x, z, fs):
    x = np.asarray(x, float); z = np.asarray(z, float)
    sig = x + 1j*z
    N = sig.size

    w = hamming_repo(N)
    xw = sig * w

    X = np.fft.fft(xw, n=N)
    absX = np.abs(X)
    mx = absX * absX

    res = float(np.dot(w, w))
    mx = mx / res

    num_unique = N // 2 + 1
    mx = mx[:num_unique]

    # one-sided scaling (x2 except DC and Nyquist)
    temp1 = mx[0]
    temp2 = mx[-1]
    mx = mx * 2.0
    mx[0] = temp1
    mx[-1] = temp2

    Pxx = mx / fs
    Fx = (np.arange(num_unique, dtype=float) * fs) / N
    return Fx, Pxx

def discard_below(Fx, Pxx, cutoff=0.05):
    Fx = np.asarray(Fx, float); Pxx = np.asarray(Pxx, float)
    idx = np.argmax(Fx >= cutoff) if np.any(Fx >= cutoff) else len(Fx)
    return Fx[idx:], Pxx[idx:]

def fn_repo(Fx, Pxx, frac):
    total = float(np.sum(Pxx))
    if total <= 0:
        return np.nan
    cur = 0.0
    i = 0
    for i in range(len(Pxx)):
        cur += float(Pxx[i])
        if cur / total > frac:
            break
    return float(Fx[i+1]) if (i + 1) < len(Fx) else float(Fx[-1])

def centroid_repo(Fx, Pxx):
    u1 = float(np.sum((Fx**1) * Pxx))
    u0 = float(np.sum((Fx**0) * Pxx))
    return float(np.sqrt(u1/u0)) if u0 > 0 else np.nan

def fdisp_repo(Fx, Pxx):
    u0 = float(np.sum((Fx**0) * Pxx))
    u1 = float(np.sum((Fx**1) * Pxx))
    u2 = float(np.sum((Fx**2) * Pxx))
    return float(np.sqrt(max(0.0, 1.0 - (u1*u1)/(u0*u2)))) if (u0 > 0 and u2 > 0) else np.nan

# ---------------- Slopes on PCA plane (your request) ----------------
def slopes_on_plane(x, z, eps=1e-12):
    """
    Define slopes as unit tangent direction of the 2D trajectory:
      slope_x[t] = dx / sqrt(dx^2 + dz^2)
      slope_z[t] = dz / sqrt(dx^2 + dz^2)
    These are in [-1,1] and can be used with asin(median(slope_*)).
    """
    x = np.asarray(x, float); z = np.asarray(z, float)
    dx = np.diff(x, prepend=x[0])
    dz = np.diff(z, prepend=z[0])
    denom = np.sqrt(dx*dx + dz*dz) + eps
    sx = dx / denom
    sz = dz / denom
    return sx, sz

# ---------------- Full extraction: Raw accel -> PCA -> features ----------------
def extract_paper_features_pca(window, fs=30.0, apply_repo_lowpass=True):
    # 1) PCA plane
    x, z, eigvals = pca_plane(window)

    # 2) 3.5 Hz low-pass (repo-style)
    if apply_repo_lowpass:
        x, z = lowpass_3p5_repo(x, z, fs=int(fs))

    # --- Time domain (paper + repo consistent) ---
    median_x = mad_like_repo(x)  # "Median X" MAD :contentReference[oaicite:10]{index=10}
    median_z = mad_like_repo(z)

    # "Median Distance" in repo uses median center
    mx = med(x); mz = med(z)
    median_dist = float(np.median(np.sqrt((x-mx)**2 + (z-mz)**2)))  # :contentReference[oaicite:11]{index=11}

    # RMS uses mean center (repo)
    cx = float(np.mean(x)); cz = float(np.mean(z))
    rms = float(np.sqrt(np.mean((x-cx)**2 + (z-cz)**2)))  # :contentReference[oaicite:12]{index=12}

    # Path (paper)
    dx = np.diff(x, prepend=x[0])
    dz = np.diff(z, prepend=z[0])
    path = float(np.sum(np.sqrt(dx*dx + dz*dz)))  # :contentReference[oaicite:13]{index=13}

    # Ellipse features
    a, b, area, perim = ellipse_ab_area_perim(x, z)
    range_a = 2.0 * a
    range_b = 2.0 * b

    # Median Frequency (paper definition) :contentReference[oaicite:14]{index=14}
    T = len(x) / fs
    median_freq = float(path / (T * perim)) if (T > 0 and perim > 0) else np.nan

    # --- “Angles” using slopes on PCA plane (NOT upright tilt) ---
    sx, sz = slopes_on_plane(x, z)
    med_lat_angle = float(np.degrees(np.arcsin(np.clip(med(sx), -1.0, 1.0))))
    med_vent_angle = float(np.degrees(np.arcsin(np.clip(med(sz), -1.0, 1.0))))

    # --- Frequency domain (repo PSD + discard <0.05 Hz) ---
    Fx, Pxx = pxx_fx_repo(x, z, fs)
    Fx, Pxx = discard_below(Fx, Pxx, cutoff=0.05)

    pwr = float(np.sum(Pxx))  # repo style (discrete sum)
    f50 = fn_repo(Fx, Pxx, 0.5)
    f95 = fn_repo(Fx, Pxx, 0.95)
    centroid = centroid_repo(Fx, Pxx)
    fdisp = fdisp_repo(Fx, Pxx)

    return {
        # Paper metrics
        "MedianX_MAD": median_x,
        "MedianZ_MAD": median_z,
        "MedianDistance": median_dist,
        "RMS": rms,
        "Path": path,
        "Area": area,
        "RangeA": range_a,
        "RangeB": range_b,
        "MedianFrequency": median_freq,
        "PWR": pwr,
        "F50": f50,
        "F95": f95,
        "CentroidFrequency": centroid,
        "FrequencyDispersion": fdisp,

        # PCA-plane slope angles (directional; not “upright tilt”)
        "MedianPlaneAngleX_deg": med_lat_angle,
        "MedianPlaneAngleZ_deg": med_vent_angle,

        # useful debugging
        "ExplainedVar_PC1": float(eigvals[0]),
        "ExplainedVar_PC2": float(eigvals[1]),
        "ExplainedVar_PC3": float(eigvals[2]),
    }
