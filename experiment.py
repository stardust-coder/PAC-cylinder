import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
from tensorpac import Pac
from tensorpac.utils import PSD
from tensorpac.signals import pac_signals_tort

from dataloader import chennu_phase, chennu_envelope, chennu_raw_onedim
from dataloader import process_ecog_data, load_marmoset_ecog
from dataloader import artificial_PAC_data_Tort
import pdb
from constant import get_eeg_filenames, get_electrode_names
import numpy as np
import pandas as pd
import model
import model.johnson_and_wehrly
import model.cylinder_bigraph
import warnings
warnings.filterwarnings("ignore")

from utils import visualize, density_plot, joint_plot
from utils import wrap_to_pi
from Gam_vM_MLE import MLE, MI, sample_from_GM

### Model
def JW3(P, A):
    ### MLE for JW cylinder(1978)
    M = model.johnson_and_wehrly.JW_cylinder3()
    ch = 60
    M.mle((P[:,ch],A[:,ch]))
    M.plot()

def JW5(P, A):
    ### MLE for JW cylinder(1978)
    # M = model.johnson_and_wehrly.Uniform_Normal()
    M = model.johnson_and_wehrly.Uniform_Gamma()
    ch = 60
    M.mle((P[:,ch],A[:,ch]))
    M.plot()

def sample_code_1():
    ###Load raw EEG data from Chennu et al.
    exp_id = 0
    patient_id = 2
    patient_state_id = 0
    ind_list, FILE_NAME_LIST = get_eeg_filenames()
    patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
    state_id = ind_list[patient_id][patient_state_id]
    out_id = f"{exp_id}_{patient_id}_{patient_state_id}_{patient_states[patient_state_id]}_{state_id}"
    raw = chennu_raw_onedim(patient_id=patient_id,state_id=state_id,ch_name="O1") 
    visualize(raw,scale=1e6)

    sf = 500
    T = 5 #(seconds)
    #TODO: ↓だとshapeがずれてなぜかうまく行かない
    # p = Pac(idpac=(6, 2, 1), f_pha=[4,8], f_amp=[25,40])  #1から順に、Mean Vector Length (MVL), Modulation Index (MI), Heights Ratio (HR), ndPAC, Phase-Locking Value (PLV), Gaussian Copula PAC (GCPAC)、see also https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html
    p = Pac(idpac=(2, 2, 1), f_pha='hres', f_amp='hres')
    print("PAC method: ", p.method)
    print("Surrogates:", p.str_surro)
    print("Normalization:", p.str_norm)

    # phases = p.filter(sf, raw, ftype='phase', n_jobs=1)
    # amplitudes = p.filter(sf, raw, ftype='amplitude', n_jobs=1)
    n_perm = 200 # > 1/0.05
    xpac = p.filterfit(sf, raw, n_perm=n_perm, n_jobs=-1).squeeze()
    print(xpac)
    pval = p.infer_pvalues(p=0.05) # get the corrected p-values
    plt.figure(figsize=(8, 4))
    plt.suptitle('theta-gamma phase amplitude coupling', fontsize=14)
    plt.subplot(1,2,1)
    p.comodulogram(xpac.mean(-1), title=p.method,cmap='turbo', vmin=0., pvalues=pval, levels=.05)
    plt.subplot(1,2,2)
    p.comodulogram(pval, title='P-values', cblabel='P-values', cmap='viridis_r', vmin=1. / n_perm,vmax=.05, over='lightgray')
    plt.axvline(x=10, color="white", linestyle="dashed", linewidth=2, alpha=0.5)
    plt.axhline(y=100, color="white", linestyle="dashed", linewidth=2, alpha=0.5)
    plt.tight_layout()
    plt.savefig("comodulogram.png")




def sample_code_2():
    from pynwb import NWBHDF5IO
    from hdmf.common.hierarchicaltable import to_hierarchical_dataframe, flatten_column_index
    import pandas as pd
    import h5py
    import numpy as np
    from tqdm import tqdm

    data = []
    for patient_id in tqdm(range(1,37)):
        try:
            filepath = f"/home/sukeda/data/Daume/sub-{patient_id}/sub-{patient_id}_ses-1_ecephys+image.nwb"
            io = NWBHDF5IO(filepath, 'r', load_namespaces = True)
            nwbfile = io.read()
            df = pd.DataFrame(np.array(nwbfile.acquisition["LFPs"].data)) #shape: (550276, 70)
            raw_lfp = df.loc[:,0].tolist()[::64][:2500] #32000Hzから500Hzに. channel 0のみを利用
            electrode_df = nwbfile.acquisition["LFPs"].electrodes.to_dataframe()
            data.append(raw_lfp)
        except:
            print(f"Error in loading patient id {patient_id}...")

    raw = np.array(data)
    visualize(raw,n_epochs=35)

    sf = 500
    T = 5 #(seconds)
    p = Pac(idpac=(2, 2, 1), f_pha='hres', f_amp='hres')
    print("PAC method: ", p.method)
    print("Surrogates:", p.str_surro)
    print("Normalization:", p.str_norm)

    n_perm = 200 # > 1/0.05
    xpac = p.filterfit(sf, raw, n_perm=n_perm, n_jobs=-1).squeeze()
    print(xpac)
    pval = p.infer_pvalues(p=0.05) # get the corrected p-values
    plt.figure(figsize=(8, 4))
    plt.suptitle('theta-gamma phase amplitude coupling', fontsize=14)
    plt.subplot(1,2,1)
    p.comodulogram(xpac.mean(-1), title=p.method,cmap='turbo', vmin=0., pvalues=pval, levels=.05)
    plt.subplot(1,2,2)
    p.comodulogram(pval, title='P-values', cblabel='P-values', cmap='viridis_r', vmin=1. / n_perm,vmax=.05, over='lightgray')
    plt.axvline(x=10, color="white", linestyle="dashed", linewidth=2, alpha=0.5)
    plt.axhline(y=100, color="white", linestyle="dashed", linewidth=2, alpha=0.5)
    plt.tight_layout()
    plt.savefig("comodulogram.png")




if __name__ == "__main__":
    # Load Marmoset ECoG with prerpocessing
    ecog_data, stimulus_onsets = load_marmoset_ecog()  
    data_lf, data_hg, phase_lf, envelope_hg, _ = process_ecog_data(
        ecog_data, 
        fs=1000,
        stimulus_onsets=stimulus_onsets, 
        baseline_start=-0.1, 
        baseline_end=0, 
        low_cut=1, 
        high_cut=30, 
        hg_low_cut=80, 
        hg_high_cut=160, 
        segment_time=(-0.1, 0.45)
    )
    # #↑ サイズは(998, 96, 550)
    
    # Visualization
    # for channel_idx in range(10):
    #     for interval_idx in range(10):
    #         # joint_plot(phase_lf[interval_idx,channel_idx,:], envelope_hg[interval_idx,channel_idx,:],name=f"interval{interval_idx}-ch{channel_idx}-joint")
    #         density_plot(phase_lf[interval_idx,channel_idx,:], envelope_hg[interval_idx,channel_idx,:],name=f"interval{interval_idx}-ch{channel_idx}-kde")

    
    # channel wise distribution. 
    # channel_idx = 0
    # density_plot(phase_lf[:,channel_idx,:].flatten()[10000:20000], envelope_hg[:,channel_idx,:].flatten()[10000:20000],name=f"ch{channel_idx}")

    ###########################   
    ### Gamma-von Mises with MLE
    channel_idx = 36
    Theta_, X_ = phase_lf[:10,channel_idx,:].flatten(), envelope_hg[:10,channel_idx,:].flatten()
    data = np.column_stack([wrap_to_pi(Theta_),X_])
    print("Data shape:",data.shape)
    # joint_plot(data[:,0],data[:,1])
    # density_plot(data[:,0],data[:,1])  

    params, ci_lower, ci_upper = MLE(data)
    observed_mi = MI(data, params)
    resampled_data = sample_from_GM(N=len(data), params=params)
    resampled_mi = MI(resampled_data, params)
    import pdb; pdb.set_trace()

    ### Gamma-von Mises with MLE (repeat)
    # comodulogram = []
    # kappas = []
    # for i in range(1,30):
    #     tmp = []
    #     tmp_kappa = []
    #     for j in range(80,160):
    #         print(i,j)
    #         data_lf_ij, data_hg_ij, phase_lf_ij, envelope_hg_ij, _ = process_ecog_data(
    #             ecog_data, 
    #             fs=1000,
    #             stimulus_onsets=stimulus_onsets, 
    #             baseline_start=-0.1, 
    #             baseline_end=0, 
    #             low_cut=i, 
    #             high_cut=i+1, 
    #             hg_low_cut=j, 
    #             hg_high_cut=j+1, 
    #             segment_time=(-0.1, 0.45)
    #         )
    #         channel_idx = 36
    #         Theta_, X_ = phase_lf_ij[:10,channel_idx,:].flatten(), envelope_hg_ij[:10,channel_idx,:].flatten()
    #         data = np.column_stack([wrap_to_pi(Theta_),X_])
    #         params = MLE(data)
    #         kappa_MLE = params[1]
    #         observed_mi = max(MI(data, params),0)
    #         # resampled_data = sample_from_GM(N=len(data), params=params)
    #         # resampled_mi = MI(resampled_data, params)
    #         tmp.append(observed_mi)
    #         tmp_kappa.append(kappa_MLE)

    #         joint_plot(data[:,0],data[:,1])
    #         import pdb; pdb.set_trace()
    #     comodulogram.append(tmp)
    #     kappas.append(tmp_kappa)
    # pd.DataFrame(comodulogram).to_csv("comodulogram.csv",index=False)
    # pd.DataFrame(kappas).to_csv("kappa_MLE.csv",index=False)

    # # === 1枚目: comodulogram ===
    # arr = pd.read_csv("comodulogram.csv").to_numpy()

    # plt.figure(figsize=(6, 5))
    # plt.imshow(np.rot90(arr), extent=[1, 30, 80, 160], origin='lower', aspect='auto')
    # plt.colorbar(label="Mutual Information")

    # # 軸設定
    # plt.xlabel("Phase frequency (Hz)")
    # plt.ylabel("Amplitude frequency (Hz)")
    # plt.xticks([1, 15, 30])
    # plt.yticks(np.linspace(80, 160, num=9))

    # plt.title("Comodulogram")
    # plt.tight_layout()
    # plt.savefig("comodulogram.png", dpi=300)
    # plt.close()

    # # === 2枚目: kappa_MLE ===
    # arr = pd.read_csv("kappa_MLE.csv").to_numpy()

    # plt.figure(figsize=(6, 5))
    # plt.imshow(np.rot90(arr), extent=[1, 30, 80, 160], origin='lower', aspect='auto')
    # plt.colorbar(label="Kappa MLE")

    # plt.xlabel("Phase frequency (Hz)")
    # plt.ylabel("Amplitude frequency (Hz)")
    # plt.xticks([1, 15, 30])
    # plt.yticks(np.linspace(80, 160, num=9))

    # plt.title("Comodulogram MLE")
    # plt.tight_layout()
    # plt.savefig("kappa_MLE.png", dpi=300)
    # plt.close()

    # import pdb;pdb.set_trace()

    ###########################    
    # ### Generalized Gamma-von Mises with EM

    # channel_idx = 35
    # Theta_, X_ = phase_lf[:100,channel_idx,:].flatten(), envelope_hg[:100,channel_idx,:].flatten()
    # # Theta_, X_ = phase_lf[:10,channel_idx,100:].flatten(), envelope_hg[:10,channel_idx,100:].flatten()

    # data = np.column_stack([np.mod(Theta_, 2 * np.pi),X_])
    # joint_plot(data[:,0],data[:,1])
    # # density_plot(data[:,0],data[:,1])
    # import pdb; pdb.set_trace()


    # def random_init_from_data(data, n_components, seed=None):
    #     """
    #     EM用の初期パラメータ (μ, κ, α, β) をランダムに初期化する関数
    #     """
    #     if seed is not None:
    #         np.random.seed(seed)

    #     thetas = data[:, 0]
    #     n = len(data)

    #     init_params = []
    #     for _ in range(n_components):
    #         # ランダムにサンプルを選んでその近傍で初期化
    #         idx = np.random.randint(0, n)
    #         mu = thetas[idx].item()
    #         kappa = np.random.uniform(0.3, 3.0)
    #         alpha = np.random.uniform(0.3, 3.0)
    #         beta = np.random.uniform(0.3, 3.0)
    #         gamma = 1.0
    #         init_params.append((mu, kappa, alpha, beta))
    #         # init_params.append((mu, kappa, alpha, beta, gamma))
    #     return init_params
    
    # params_init = random_init_from_data(data,n_components=1)
    # from Gam_vM_EM import EMAlgorithm, evaluate_fit
    # # from GGam_vM_EM import EMAlgorithm, evaluate_fit #Generalizedは安定性に欠ける.
    # model = EMAlgorithm(params_init,len(params_init))
    # model.fit(data)
    # print(evaluate_fit(model,data)) #Fit具合を可視化