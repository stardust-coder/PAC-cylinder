import numpy as np
from scipy.special import gammaln
from scipy.signal import butter, sosfiltfilt, hilbert, decimate
import pandas as pd
import statsmodels.api as sm
from model.copulas import *
from model.GLMCFC import *
from model.CLL import *
from dataloader import load_nadalin_rodent
from dataloader import hilbert_phase, hilbert_envelope, bandpass_filter
import matplotlib.pyplot as plt
import seaborn as sns
from utils import MVL, Pearson

from trivariate_sim_comparison import *

def plot_func(my_df):
    sns.jointplot(
        data=my_df,
        x="theta",
        y="x1",
        kind="scatter",   # "kde" や "hex" に変えると密度表示も可
        marginal_kws=dict(bins=40, fill=True, color="gray", alpha=0.6)
    )
    plt.xlabel("phase")
    plt.ylabel("amplitude")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"theta-x1.png")

    sns.jointplot(
        data=my_df,
        x="theta",
        y="x2",
        kind="scatter",   # "kde" や "hex" に変えると密度表示も可
        marginal_kws=dict(bins=40, fill=True, color="gray", alpha=0.6)
    )
    plt.xlabel("phase")
    plt.ylabel("amplitude")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"theta-x2.png")

    plot_tail_dependence_by_theta(my_df["theta"],my_df["x1"],my_df["x2"], nbins=6, q=0.9, fname="tail_upper.png",tail='upper')
    plot_tail_dependence_by_theta(my_df["theta"],my_df["x1"],my_df["x2"], nbins=6, q=0.9, fname="tail_lower.png",tail='lower')
    

if __name__ == "__main__":
    #sampling rate of this data is 30000Hz. 
    #Human Cortex.
    #Two Patients, A (45yo, male) and B (32yo, male).
    #Each time series data is size (8503673, 1) and (6632746, 1).
    data = load_nadalin_rodent() 
    import pdb; pdb.set_trace()
    fs_raw = 30000
    key = [int(patientA.shape[0]/2)-fs_raw*10*12,int(patientA.shape[0]/2)-fs_raw*10*1, int(patientA.shape[0]/2)+fs_raw*10*4]
    key_id = 1
    raw = patientA.T[:,key[key_id]:key[key_id]+fs_raw*20][0]/1000 #mV

    def lowpass(data, cutoff, fs, order=4):
        sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
        return sosfiltfilt(sos, data)
    
    raw_lp = lowpass(raw, cutoff=300, fs=fs_raw) # 1) ローパスでLFPレンジに制限

    fs_lfp = 1000 # 2) ダウンサンプル（30000 → 1000 Hz）
    factor = fs_raw // fs_lfp  # 30000 // 1000 = 30
    lfp = decimate(raw_lp, factor, ftype='fir', zero_phase=True)
    print(f"Fs is reduced to {fs_lfp}.")

    data_lf = bandpass_filter(lfp, low_cut=4, high_cut=7, fs=fs_lfp)  # LF band-pass 
    data_hg = bandpass_filter(lfp, low_cut=100, high_cut=140, fs=fs_lfp)  # HG band-pass
    X1 = hilbert_envelope(data_lf)
    X2 = hilbert_envelope(data_hg)
    theta = hilbert_phase(data_lf)
    mu_true = np.mean(theta)
    theta = (theta - mu_true + np.pi) % (2*np.pi) - np.pi     # mu=0前提に合わせる
    real_data = np.column_stack((theta, X1, X2))
    real_data = real_data[::10,:]
    my_df = pd.DataFrame(real_data, columns=["theta", "x1", "x2"])
    print("Dataset size: ",my_df.shape)

    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot([i/fs_raw for i in range(patientA.shape[0])], patientA,  label="Patient A",color="blue")
    # ax1.plot([i/fs_lfp for i in range(patientB.shape[0])], patientB, label="Patient B",color="orange")
    for k in key:
        ax1.axvline(x=k/fs_raw, color='red', linestyle='--', linewidth=1.5)
        ax1.axvline(x=(k+fs_raw*20)/fs_raw, color='red', linestyle='--', linewidth=1.5)

    ax1.set_ylabel("Voltage (μV)")
    ax1.set_xlabel("time (second)")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("human_seizure.png")
    plt.close(fig)

    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot([i/fs_raw for i in range(raw.shape[0])], raw,  label="Patient A",color="blue")
    ax1.set_ylabel("Voltage (μV)")
    ax1.set_xlabel("time (second)")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("human_seizure_extracted.png")
    plt.close(fig)

    plot_func(my_df)

    my_copula = t_c_phi_wrapper

    def fit_func_full(df, **kwargs):
        data = df[["theta", "x1", "x2"]].values
        (params, phi_hat) = MLE(data, copula_func=my_copula)
        kappa, alpha, beta1, beta2 = params
        return dict(kappa=kappa, alpha=alpha, beta=(beta1, beta2), phi=phi_hat)

    def fit_func_null(df, **kwargs):
        # φ固定=0（独立コピュラ）で同じ推定
        data = df[["theta", "x1", "x2"]].values
        (params, _) = MLE(data, copula_func=lambda u,v,phi: 1.0)
        kappa, alpha, beta1, beta2 = params
        return dict(kappa=kappa, alpha=alpha, beta=(beta1, beta2), phi=0.0)
    
    def fit_func_null_no_phase(df, **kwargs):
        data  = df[["theta","x1","x2"]].values
        theta = data[:,0]; x1 = data[:,1]; x2 = data[:,2]
        (params, phi_hat) = MLE(data, copula_func=my_copula)
        _, alpha_hat, _, _ = params
        kappa0 = 0.0
        def beta_mle_closed_form(theta, x, alpha, kappa, eps=1e-12):
            s = 1.0 - np.tanh(kappa)*np.cos(theta)   # = 1.0 when kappa=0
            mu = np.mean(x * s)
            return float(alpha / max(mu, eps))
        
        beta1_0 = beta_mle_closed_form(theta, x1, alpha_hat, kappa0)
        beta2_0 = beta_mle_closed_form(theta, x2, alpha_hat, kappa0)

        def nll_phi(phi): # φ は自由推定
            return -log_likelihood_pred_model(
                data, my_copula, kappa0, alpha_hat, (beta1_0, beta2_0), phi
            )
        from scipy.optimize import minimize
        res = minimize(lambda p: nll_phi(p), x0=np.array([0.0]), method="L-BFGS-B")
        phi_hat = float(res.x)

        return dict(kappa=kappa0, alpha=alpha_hat, beta=(beta1_0, beta2_0), phi=phi_hat)

    def loglik_func(df, kappa, alpha, beta, phi, **kwargs):
        data = df[["theta", "x1", "x2"]].values

        def _condata(i):
            nbins = 6
            edges = np.linspace(-np.pi, np.pi, nbins+1)
            begin,end = edges[i], edges[i+1]
            return df.query("@begin <= theta < @end").to_numpy()

        if type(phi) == list:
            assert len(phi) == 6
            L_ = [log_likelihood_pred_model(_condata(i), my_copula, kappa, alpha, beta, x) for i,x in enumerate(phi)]
            print(L_)
            return sum(L_)
        else:
            return log_likelihood_pred_model(data, my_copula, kappa, alpha, beta, phi)


    res_ours = run_fit_and_metrics_general(
        df=my_df,
        model_type="custom",
        fit_func_full=fit_func_full,
        fit_func_null=fit_func_null,
        loglik_func=loglik_func
    )

    res = res_ours
    print(f"AIC_full={res['AIC_full']:.3f}, AIC_null={res['AIC_null']:.3f}, ΔAIC={res['Delta_AIC_null_to_full']:.3f}")

    print("MVL(low phase vs low amplitude)=", MVL(real_data[:,0],real_data[:,1]))
    print("MVL(low phase vs high amplitude)=", MVL(real_data[:,0],real_data[:,2]))
    print("Pearson's correlation coefficient=", Pearson(real_data[:,1],real_data[:,2]))

    (kappa_hat, alpha_hat, beta1_hat, beta2_hat) = (res["res_full"]["kappa"],res["res_full"]["alpha"],res["res_full"]["beta"][0],res["res_full"]["beta"][1])
    mle_tuple = (kappa_hat, alpha_hat, beta1_hat, beta2_hat)
    phi_hat = res["res_full"]["phi"]
    
    
    _, I_x1x2_given_theta = negative_copula_entropy_from_MLE_tuple(
        data=real_data,
        copula_func=my_copula,
        mle_tuple=mle_tuple,
        phi_hat=phi_hat
    )
    print("I(X1; X2 | θ) =", I_x1x2_given_theta)

    # Θ–X1
    I_theta_x1 = mutual_information_theta_x1_from_cyl_model(
        theta=real_data[:,0],
        x1=real_data[:,1],
        log_f_joint=log_f_joint_theta_x1,
        log_f_theta_marg=log_f_theta_marg,
        log_f_x1_marg=log_f_x1_marg,
        params_joint=(kappa_hat, alpha_hat, beta1_hat),
        params_theta_marg=(kappa_hat, alpha_hat),
        params_x1_marg=(kappa_hat, alpha_hat, beta1_hat)
    )

    # Θ–X2
    I_theta_x2 = mutual_information_theta_x2_from_cyl_model(
        theta=real_data[:,0],
        x2=real_data[:,2],
        log_f_joint=log_f_joint_theta_x2,
        log_f_theta_marg=log_f_theta_marg,
        log_f_x2_marg=log_f_x2_marg,
        params_joint=(kappa_hat, alpha_hat, beta2_hat),
        params_theta_marg=(kappa_hat, alpha_hat),
        params_x2_marg=(kappa_hat, alpha_hat, beta2_hat)
    )

    print("I(Θ; X1) =", I_theta_x1)
    print("I(Θ; X2) =", I_theta_x2)

    I_theta_x1x2 = I_theta_x1 + I_theta_x2 + I_x1x2_given_theta     # ← 真の三変量MI
    I_theta_x2_given_x1 = I_theta_x2 + I_x1x2_given_theta            # PAC指標
    I_theta_x1_given_x2 = I_theta_x1 + I_x1x2_given_theta
    print("I(θ, x1, x2) =", I_theta_x1x2)
    print("I(Θ; X2 | X1) =", I_theta_x2_given_x1)
    print("I(Θ; X1 | X2) =", I_theta_x1_given_x2)