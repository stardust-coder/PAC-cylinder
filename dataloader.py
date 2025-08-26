import sys
sys.path.append(".")
from constant import get_eeg_filenames, get_electrode_names
import utils
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import pandas as pd
import numpy as np

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