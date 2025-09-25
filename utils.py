import numpy as np
import scipy
import matplotlib.pyplot as plt

# ヒルベルト変換
def hilbert_transform(signal, dt=1e-4):
    '''
    Input ;
        signal : np.array with shape (len,)
    Return ; 
        z :  複素数信号. zの実部はsignal, zの虚部はヒルベルト変換された信号.
        env : envelope, zの絶対値
        phase_inst : 瞬時位相
        freq_inst : 瞬時周波数
    '''
    z = scipy.signal.hilbert(signal)
    env = np.abs(z)
    phase_inst = np.angle(z)
    freq_inst = np.gradient(phase_inst)/dt/(2.0*np.pi)
    return z, env, phase_inst, freq_inst

###############################################################
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Function to apply band-pass filter
def bandpass_filter(data, low_cut, high_cut, fs):
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# Function to calculate the common median reference (CMR)
def common_median_reference(data):
    median_reference = np.median(data, axis=0)
    return data - median_reference

# Function to calculate the envelope using Hilbert transform
def hilbert_envelope(data):
    analytic_signal = hilbert(data, axis=-1)
    envelope = np.abs(analytic_signal)
    return envelope

# Function to baseline correct the data
def baseline_correct(data, baseline_start, baseline_end, fs):
    baseline_samples = int(baseline_start * fs), int(baseline_end * fs)
    baseline_mean = np.mean(data[:, baseline_samples[0]:baseline_samples[1]], axis=-1)
    return data - baseline_mean[:, None]

# Function to remove outlier segments
def remove_outliers(data, threshold=3):
    # Calculate standard deviation for each segment
    std_per_segment = np.std(data, axis=-1)
    # Calculate overall standard deviation
    overall_std = np.std(data)
    # Identify outlier segments
    outlier_indices = np.where(std_per_segment > threshold * overall_std)[0]
    # Remove outliers
    data_clean = np.delete(data, outlier_indices, axis=0)
    
    return data_clean

# Assume data is a 2D numpy array (channels x time) and fs is the sampling rate
def process_ecog_data(data, fs, baseline_start=-0.1, baseline_end=0, low_cut=1, high_cut=30, hg_low_cut=80, hg_high_cut=160, segment_time=(-0.1, 0.45)):
    # Step 1: Re-reference data using CMR
    data_cmr = common_median_reference(data)

    # Step 2: Apply band-pass filtering
    data_lf = bandpass_filter(data_cmr, low_cut, high_cut, fs)  # LF band-pass (1-30 Hz)
    data_hg = bandpass_filter(data_cmr, hg_low_cut, hg_high_cut, fs)  # HG band-pass (80-160 Hz)

    # Step 3: Calculate the envelope of the HG frequency band
    envelope_hg = hilbert_envelope(data_hg)

    # Step 4: Segment data based on the time window around the stimulus
    # Assume we have an array of stimulus onsets (in samples)
    segment_start = int(segment_time[0] * fs)
    segment_end = int(segment_time[1] * fs)
    
    # Create segments based on stimulus onsets (assuming stimulus_onsets is an array of sample indices)
    # You need to adjust this part according to how you have the stimulus information
    # For example:
    stimulus_onsets = np.array([500, 1500, 2500])  # Example stimulus onsets (in samples)
    segments = [data[:, onset+segment_start:onset+segment_end] for onset in stimulus_onsets]

    # Concatenate segments into a 3D array (trials x channels x time)
    data_segments = np.stack(segments, axis=0)

    # Step 5: Remove outliers
    data_clean = remove_outliers(data_segments)

    # Step 6: Apply baseline correction
    data_baseline_corrected = baseline_correct(data_clean, baseline_start, baseline_end, fs)

    # Return the processed data (LF, HG, and envelope for HG)
    return data_lf, data_hg, envelope_hg, data_baseline_corrected



def visualize(data, n_epochs=10, scale=1):
    import matplotlib.pyplot as plt
    T = 5
    time = [i/500 for i in range(2500)]
    data = data * scale

    plt.figure(figsize=(8, 4))
    plt.subplot(1,2,1)
    plt.title(str(n_epochs)+" trials")
    plt.imshow(data, cmap="turbo", extent=(0,T,0,n_epochs), aspect='auto')
    plt.xlim(0, T); plt.xlabel("Time (sec)"); plt.ylabel("Trials")
    plt.subplot(1,2,2)
    plt.title("Example : five trials")
    for i in range(5):
        plt.plot(time, data[i]*0.02+i+0.75)
    plt.xlim(0, T); plt.xlabel("Time (sec)"); plt.ylabel("Trials")
    plt.tight_layout()
    plt.savefig("visualize_five_epochs.png")

def density_plot(x,y,name="density_plot"):
    from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(8,6))
    plt.rcParams["font.size"] = 18
    ax=fig.add_subplot(111)
    # KDE probability
    # import pdb; pdb.set_trace()
    xy = np.vstack([x,y]) 
    z = gaussian_kde(xy)(xy)
    # zの値で並び替え→x,yも並び替える
    idx = z.argsort() 
    x, y, z = x[idx], y[idx], z[idx]
    im = ax.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)
    #軸設定
    ax.set_xlabel("Phase")
    ax.set_ylabel("Amplitude")
    # ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi*2)
    # ax.set_xlim(-np.pi, np.pi)
    plt.show()
    plt.savefig(f"output/{name}.png")
    print(f"KDE density plot saved to output/{name}.png !")

def joint_plot(x,y,name="joint_plot"):
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame({"Phase":x, "Amplitude":y})
    sns.jointplot(data=df, x="Phase", y="Amplitude")
    plt.savefig(f"output/{name}.png")
    print(f"Joint plot saved to output/{name}.png !")

def wrap_to_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def _rank_average(x: np.ndarray) -> np.ndarray:
    """
    1D 配列 x の平均順位（ties は平均）を 1..n の実数で返す。
    SciPy/pandas なしで動く実装。
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")          # 安定ソート
    ranks = np.empty(n, dtype=float)
    xs = x[order]
    # 同値ブロックの境界を見つける
    boundaries = np.concatenate(([0], np.nonzero(np.diff(xs))[0] + 1, [n]))
    # 各ブロックに平均順位を付与
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        # 1-based rank の範囲: [b0+1, b1]
        avg = (b0 + 1 + b1) / 2.0
        ranks[order[b0:b1]] = avg
    return ranks

def rank_uniform_2d(X: np.ndarray, *, a: float = 0.0, method: str = "average",
                    seed: int | None = None) -> np.ndarray:
    """
    Nx2 の実数配列 X を、列ごとに rank-based に [0,1] へ写像。
    返り値 U は Nx2 で、各列が (ほぼ) Unif(0,1) マージナル。

    パラメータ
    ----------
    X : (N,2) array-like
        入力データ（NaN は不可。必要なら事前に除去/補完してください）
    a : float, default 0.0
        正規化 u = (rank - a) / (n + 1 - 2a) の a。
        代表値: a=0     -> Weibull (rank/(n+1))
               a=0.5   -> Hazen  ((rank-0.5)/n)
               a=3/8   -> Blom   ((rank-3/8)/(n+1/4))
    method : {"average","random"}, default "average"
        ties の扱い。
        - "average": 同値への平均順位（推奨）
        - "random" : ties を微小ノイズでランダムに解消して通常順位にする
    seed : int | None
        method="random" の乱数シード

    戻り値
    ------
    U : (N,2) ndarray
        各列が [0,1] に写像されたデータ
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X は形状 (N,2) の配列である必要があります。")
    if np.isnan(X).any():
        raise ValueError("X に NaN が含まれています。除去/補完してください。")

    n = X.shape[0]
    if method not in {"average", "random"}:
        raise ValueError("method は 'average' または 'random'。")

    U = np.empty_like(X, dtype=float)

    if method == "average":
        r1 = _rank_average(X[:, 0])
        r2 = _rank_average(X[:, 1])
    else:  # random
        rng = np.random.default_rng(seed)
        # 同値を壊すために非常に小さなノイズを加えて通常順位
        eps1 = (np.finfo(float).eps ** 0.5) * rng.standard_normal(n)
        eps2 = (np.finfo(float).eps ** 0.5) * rng.standard_normal(n)
        r1 = np.argsort(np.argsort(X[:, 0] + eps1, kind="mergesort"), kind="mergesort").astype(float) + 1.0
        r2 = np.argsort(np.argsort(X[:, 1] + eps2, kind="mergesort"), kind="mergesort").astype(float) + 1.0

    denom = (n + 1.0 - 2.0 * a)
    U[:, 0] = (r1 - a) / denom
    U[:, 1] = (r2 - a) / denom

    # 数値誤差でわずかに [0,1] を超えるのを防止
    U = np.clip(U, 0.0, 1.0)
    return U

def scatter_unit_square(U: np.ndarray, s: float = 8.0, filename: str = "scatter_unit_square.png"):
    """
    rank_uniform_2d の出力 U (N,2) を単位正方形に散布図表示。
    """
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError("U は形状 (N,2) の配列である必要があります。")
    plt.figure(figsize=(5, 5))
    plt.scatter(U[:, 0], U[:, 1], s=s)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("U1"); plt.ylabel("U2"); plt.title("Rank-based scatter on [0,1]^2")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(filename)
