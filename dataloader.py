import numpy as np
from scipy import signal
import sys
sys.path.append(".")
from constant import get_eeg_filenames, get_electrode_names
import utils
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import pandas as pd

ind_list, FILE_NAME_LIST = get_eeg_filenames()

def chennu_phase(patient_id,state_id,dim, freq_bottom, freq_top, epochs = [0]):
    # Hz α：8-15, β: 12-25, γ: 25-40

    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
    PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm"
        )
        return data

    def get_instantaneous_phase(signal, start, end, verbose=False):
        """get instantaneous phase"""
        _, _, phase, _ = utils.hilbert_transform(signal=signal)
        return phase

    loaded_eeg = load_human_eeg(PATH_TO_DATA)
    loaded_eeg_filt = loaded_eeg.copy().filter(freq_bottom, freq_top)
    raw_eeg = loaded_eeg_filt.get_data(copy=True)
    
    '''
    ### Settings
    '''
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    num_electrodes = 91

    '''
    ### Data preprocessing
    '''
    ###Phase
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_instantaneous_phase(
                raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    phase_data_arr = data_df.to_numpy()

    ### Extract target electrode channels
    montage = loaded_eeg.get_montage()
    main_electrodes = []
    
    for item in get_electrode_names(dim): #choose dim from {5,9,61,91}
        main_electrodes.append(montage.ch_names.index(item))
    
    phase_data_arr = phase_data_arr[:, main_electrodes] 
    return phase_data_arr


def chennu_envelope(patient_id,state_id,dim, freq_top, freq_bottom, epochs = [0]):
    # Hz α：8-15, β: 12-25, γ: 25-40

    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
    PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm"
        )
        return data
    
    def get_envelope(signal, start, end, verbose=False):
        """get amplitude of power"""
        _, env, _, _ = utils.hilbert_transform(signal=signal)
        return env
    
    loaded_eeg = load_human_eeg(PATH_TO_DATA)
    loaded_eeg_filt = loaded_eeg.copy().filter(freq_bottom, freq_top)
    raw_eeg = loaded_eeg_filt.get_data(copy=True)

    '''
    ### Settings
    '''
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    num_electrodes = 91

    '''
    ### Data preprocessing
    '''
    ###Amplitude
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_envelope(
                raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    power_data_arr = data_df.to_numpy()

    ### Extract target electrode channels
    montage = loaded_eeg.get_montage()
    main_electrodes = []
    
    for item in get_electrode_names(dim): #choose dim from {5,9,61,91}
        main_electrodes.append(montage.ch_names.index(item))
    
    power_data_arr = power_data_arr[:, main_electrodes]
    return power_data_arr

def chennu_raw_onedim(patient_id,state_id,ch_name="C3",epochs=list(range(10))):
    # Hz α：8-15, β: 12-25, γ: 25-40

    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
    PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm"
        )
        return data

    loaded_eeg = load_human_eeg(PATH_TO_DATA)
    raw_eeg = loaded_eeg.get_data(copy=True)

    '''
    ### Settings
    '''
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    num_electrodes = 91

    ### Extract target electrode channels
    montage = loaded_eeg.get_montage()
    idx = get_electrode_names(num_electrodes).index(ch_name) #dim=91のうちのdim番目

    print("Loading...")
    print("Electrode name:", ch_name)
    print("Electrode index:", idx)
    df_list = []
    for epoch in epochs:
        data_df = list(raw_eeg[epoch][idx - 1])
        df_list.append(data_df)
    return np.array(df_list)


def load_chennu(patient_id=2, patient_state_id=0, dim=61):
    ind_list, FILE_NAME_LIST = get_eeg_filenames()
    patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
    state_id = ind_list[patient_id][patient_state_id]
    print(f"Loading Human EEG (Chennu), PatientID:{patient_id},StateID:{patient_state_id},State:{patient_states[patient_state_id]}")

    # delta_phase = chennu_phase(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=0.5,freq_top=4)  #(2500, 61)
    # delta_amp = chennu_envelope(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=0.5,freq_top=4) #(2500, 61)
    theta_phase = chennu_phase(patient_id=patient_id,state_id=state_id,dim=dim,freq_bottom=4,freq_top=8) 
    # theta_amp = chennu_envelope(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=4,freq_top=8) 
    # alpha_phase = chennu_phase(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=8,freq_top=15) 
    # alpha_amp = chennu_envelope(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=8,freq_top=15) 
    # beta_phase = chennu_phase(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=12,freq_top=25) 
    # beta_amp = chennu_envelope(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=12,freq_top=25) 
    # gamma_phase = chennu_phase(patient_id=patient_id,state_id=state_id,dim=61,freq_bottom=25,freq_top=40) 
    gamma_amp = chennu_envelope(patient_id=patient_id,state_id=state_id,dim=dim,freq_bottom=25,freq_top=40) 

    # phase_data_arr = np.concatenate([delta_phase, theta_phase, alpha_phase, beta_phase, gamma_phase])
    # amp_data_arr = np.concatenate([delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp])
    return theta_phase, gamma_amp


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
def hilbert_phase(data):
    analytic_signal = hilbert(data, axis=-1)
    phase = np.angle(analytic_signal)
    return phase

def hilbert_envelope(data):
    analytic_signal = hilbert(data, axis=-1)
    envelope = np.abs(analytic_signal)
    return envelope

# Function to baseline correct the data
def baseline_correct(data, baseline_start, baseline_end, fs):
    baseline_samples = int(baseline_start * fs), int(baseline_end * fs)
    baseline_mean = np.mean(data[:, :, :-baseline_samples[0]], axis=-1)
    
    return data - baseline_mean[:, :, None]

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
def process_ecog_data(data, fs, stimulus_onsets, baseline_start=-0.1, baseline_end=0, low_cut=1, high_cut=30, hg_low_cut=80, hg_high_cut=160, segment_time=(-0.1, 0.45)):
    # Step 4: Segment data based on the time window around the stimulus
    # Assume we have an array of stimulus onsets (in samples)
    segment_start = int(segment_time[0] * fs)
    segment_end = int(segment_time[1] * fs)
    
    # Create segments based on stimulus onsets (assuming stimulus_onsets is an array of sample indices)
    # You need to adjust this part according to how you have the stimulus information
    # For example:
    # stimulus_onsets = np.array([500, 1500, 2500])  # Example stimulus onsets (in samples)
    segments = [data[:, onset+segment_start:onset+segment_end] for onset in stimulus_onsets]

    # Concatenate segments into a 3D array (trials x channels x time)
    data_segments = np.stack(segments, axis=0)

    # Step 5: Remove outliers
    data_clean = remove_outliers(data_segments)

    
    # Step 6: Apply baseline correction
    data_baseline_corrected = baseline_correct(data_clean, baseline_start, baseline_end, fs)


    # Step 1: Re-reference data using CMR
    data_cmr = common_median_reference(data_baseline_corrected)

    # Step 2: Apply band-pass filtering
    data_lf = bandpass_filter(data_cmr, low_cut, high_cut, fs)  # LF band-pass (1-30 Hz)
    data_hg = bandpass_filter(data_cmr, hg_low_cut, hg_high_cut, fs)  # HG band-pass (80-160 Hz)

    phase_lf = hilbert_phase(data_lf)

    # Step 3: Calculate the envelope of the HG frequency band
    envelope_hg = hilbert_envelope(data_hg)

    # Return the processed data (LF, HG, and envelope for HG)
    return data_lf, data_hg, phase_lf, envelope_hg, data_baseline_corrected

def load_marmoset_ecog(name="Ji",ind=1):
    import scipy
    def load_marmoset_ecog_raw(name,ind):
        length = -1
        if name == "Ji":
            assert ind in [1,2,3,5,15]
            res = []
            for chan in range(1,97):
                info = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Ji20180308S{ind}c/ECoG_ch{chan}.mat')
                data = info['ECoGData'][:,:length]
                res.append(data)
            event = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Ji20180308S{ind}c/Event.mat')
            onsets = event["cntEvent"][:,5].astype(int)
            return np.concatenate(res,axis=0), onsets
        if name == "Or":
            assert ind in [2,3,4,6,16]
            res = []
            for chan in range(1,97):
                data = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Or20171207S{ind}c/ECoG_ch{chan}.mat')['ECoGData'][:,:length]
                res.append(data)
            event = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Or20171207S{ind}c/Event.mat')
            onsets = event["cntEvent"][:,5].astype(int)
            return np.concatenate(res,axis=0), onsets
    
    raw, onsets = load_marmoset_ecog_raw(name,ind) #(96, 508864)
    return raw, onsets


def artificial_PAC_data_Tort():
    from tensorpac.signals import pac_signals_tort
    n_epochs = 20    # number of trials
    sf = 1000.        # sampling frequency
    T = 0.5          # one trials time (sec)
    n_times = sf * T # number of time points

    # Create artificially coupled signals using Tort method :
    data, time = pac_signals_tort(f_pha=10, f_amp=100, noise=2, n_epochs=n_epochs, 
                                dpha=10, damp=10, sf=sf, n_times=n_times)
    return data


# ---------------------------
# 1/f^alpha ピンクノイズ
# ---------------------------
def make_pink_noise(alpha: float, L: int, dt: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    MATLAB: make_pink_noise(alpha,L,dt)
    FFT の振幅スペクトルを |f|^{-alpha} で整形してから逆FFTして 1/f^alpha ノイズを作る。
    """
    if rng is None:
        rng = np.random.default_rng()

    x = rng.standard_normal(L)
    xf = np.fft.fft(x)
    A = np.abs(xf)
    phase = np.angle(xf)

    # 周波数軸（両側）。絶対値にして |f|^{-alpha} を適用。
    f = np.abs(np.fft.fftfreq(L, d=dt))
    # f=0 の発散回避
    with np.errstate(divide="ignore"):
        one_over_f = 1.0 / (f ** alpha)
    one_over_f[0] = 0.0

    Anew = A * one_over_f
    xf_new = Anew * np.exp(1j * phase)
    x_new = np.fft.ifft(xf_new).real
    return x_new


def nadalin(pac_mod: float,
           aac_mod: float,
           sim_method: str,
           rng: np.random.Generator | None = None):
    """
    MATLAB: [XX,P,Vlo,Vhi,t] = simfun(pac_mod,aac_mod,sim_method,pval,ci,AIC[,q])

    pac_mod : PAC 強度（高周波包絡を低周波ピーク同期のハン窓で変調）
    aac_mod : AAC 強度（高/低の振幅–振幅結合の強さ）
    sim_method : 'GLM' | 'pink' | 'spiking'
    pval, ci, AIC : GLM 推定側のオプション（ここでは受け渡しだけ）
    q : 省略可。GLM 実装側にそのまま渡す
    rng : numpy.random.Generator（再現性が欲しいときに指定）

    戻り値:
      XX, P : GLM 出力（ここではスタブ）。実装があれば差し替えてください。
      Vlo, Vhi : 低/高周波バンド信号（最終的に観測信号から再抽出したもの）
      t : 時間軸（秒）
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- シミュレーション条件
    dt = 0.002
    Fs = 1.0 / dt
    fNQ = Fs / 2.0
    N = int(20 / dt + 4000)  # 端のフィルタ歪みを後で捨て、正味 20s にする

    # ---- 低周波バンドを作る（ピンクノイズ→FIRLS→filtfilt）
    Vpink = make_pink_noise(1.0, N, dt, rng)
    Vpink -= Vpink.mean()

    def _firls_bandpass(locut, hicut, order_scale):
        # MATLAB の firls(order, f, m) は「次数」指定。SciPy の firls は「タップ数」指定なので +1。
        filtorder = int(order_scale * (Fs / locut))
        if filtorder % 2 != 0:
            filtorder += 1
        numtaps = filtorder + 1
        trans = 0.15
        bands = [0,
                 (1 - trans) * (locut / fNQ),
                 (locut / fNQ),
                 (hicut / fNQ),
                 (1 + trans) * (hicut / fNQ),
                 1.0]
        desired = [0, 0, 1, 1, 0, 0]
        b = signal.firls(numtaps, bands, desired)
        a = 1.0
        return b, a

    # Low: 
    bL, aL = _firls_bandpass(4.0, 7.0, order_scale=3)
    Vlo = signal.filtfilt(bL, aL, Vpink)

    # ---- 高周波バンド用のピンクノイズを作り直し
    Vpink = make_pink_noise(1.0, N, dt, rng)
    Vpink -= Vpink.mean()

    # High: 
    bH, aH = _firls_bandpass(100.0, 140.0, order_scale=10)
    Vhi = signal.filtfilt(bH, aH, Vpink)

    # ---- フィルタ端の 4 s を両端で捨てる（MATLAB: 2001:end-2000）
    Vlo = Vlo[2000:-2000]
    Vhi = Vhi[2000:-2000]
    t = np.arange(1, Vlo.size + 1) * dt
    N = Vlo.size

    # ---- 低周波ピークに同期した変調窓 s(t) を作る（ハン窓 21 点）
    peaks, _ = signal.find_peaks(Vlo)
    AmpLo = np.abs(signal.hilbert(Vlo))  # 低周波の包絡
    s = np.zeros_like(Vhi)
    for idx in peaks:
        if 10 < idx < (Vhi.size - 10):
            s[idx - 10: idx + 11] = np.hanning(21)
    s /= (s.max() + np.finfo(float).eps)  # 0–1 正規化

    # ---- sim_method 切り替え
    if sim_method == 'GLM':
        # Ahi(t) = 1 + pac_mod * s(t)
        Ahi = 1.0 + pac_mod * s
        # φ_hi は元の高周波から Hilbert 位相を抽出し、振幅 Ahi でコサイン化
        phi_hi = np.angle(signal.hilbert(Vhi))
        Vhi = 0.01 * Ahi * np.cos(phi_hi)
        # AAC: 高/低の振幅相関（AmpLo の正規化でスケーリング）
        Vhi = Vhi * (1.0 + aac_mod * AmpLo / AmpLo.max())

    elif sim_method == 'pink':
        # 高周波の包絡をピーク同期窓で AM し、さらに低周波包絡で AM
        Vhi = Vhi * (1.0 + pac_mod * s)
        Vhi = Vhi * (1.0 + aac_mod * AmpLo / AmpLo.max())

    elif sim_method == 'spiking':
        # スパイキング・プロセスで高周波を生成（MATLAB と同じ式）
        N = int(20 / dt)
        t = np.arange(1, N + 1) * dt
        Alo = 1.0 + (np.sin(2 * np.pi * t * 0.1) + 1.0) / 2.0
        Philo = np.pi * signal.sawtooth(2 * np.pi * t * 4)  # [-π, π]三角波
        Vlo = Alo * np.cos(Philo)

        Philo_star = np.pi + Alo * np.pi
        sigma = 0.01
        tri = signal.sawtooth(Philo - Philo_star, width=0.5)  # [-1, 1] の三角波
        lam = (1.0 / np.sqrt(2 * np.pi * sigma)) * np.exp(-(1.0 + tri) ** 2 / (2 * sigma ** 2))
        lam = 0.001 + 0.3 * lam / lam.max()
        Vhi = rng.binomial(1, lam).astype(float)

        Vlo = Vlo + 0.1 * rng.standard_normal(Vlo.size)
        Vhi = Vhi + 0.1 * rng.standard_normal(Vhi.size)

    else:
        raise ValueError("sim_method must be 'GLM', 'pink', or 'spiking'.")

    # ---- 観測信号を作ってから、もう一度バンド分け（spiking 以外）
    if sim_method != 'spiking':
        Vpink2 = make_pink_noise(1.0, N, dt, rng)
        noise_level = 0.01
        V1 = Vlo + Vhi + noise_level * Vpink2

        # Low:（再び抽出）
        Vlo = signal.filtfilt(bL, aL, V1)
        # High: （再び抽出）
        Vhi = signal.filtfilt(bH, aH, V1)

    return V1, Vlo, Vhi, t
