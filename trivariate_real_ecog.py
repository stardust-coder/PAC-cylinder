import numpy as np
from model.copulas import *
from model.GLMCFC import *
from model.CLL import MLE, log_likelihood_pred_model
from dataloader import process_ecog_data, load_marmoset_ecog, hilbert_envelope
import matplotlib.pyplot as plt
import seaborn as sns
from trivariate_sim_comparison import *

if __name__ == "__main__":
    ecog_data, stimulus_onsets = load_marmoset_ecog()  
    print("ECoG Data shape:", ecog_data.shape)
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
    #サイズは(998, 96, 550)
    channel_ = 90
    epoch_ = np.array([i for i in range(30)])
    theta = phase_lf[epoch_, channel_, ::10].reshape(-1)
    x1_env = hilbert_envelope(data_lf[epoch_, channel_, :])   # 形は (len(epoch_), 550)
    x1 = x1_env[:, ::10].reshape(-1)
    x2 = envelope_hg[epoch_, channel_, ::10].reshape(-1)
    my_df = pd.DataFrame({"theta": theta, "x1": x1, "x2": x2})

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
    plt.savefig(f"ecog-ch{channel_}-x1.png")

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
    plt.savefig(f"ecog-ch{channel_}-x2.png")
    # import pdb; pdb.set_trace()

    my_copula = clayton_c

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
                data, gaussian_c, kappa0, alpha_hat, (beta1_0, beta2_0), phi
            )
        from scipy.optimize import minimize
        res = minimize(lambda p: nll_phi(p), x0=np.array([0.0]), method="L-BFGS-B")
        phi_hat = float(res.x)

        return dict(kappa=kappa0, alpha=alpha_hat, beta=(beta1_0, beta2_0), phi=phi_hat)

    def loglik_func(df, kappa, alpha, beta, phi, **kwargs):
        data = df[["theta", "x1", "x2"]].values
        return log_likelihood_pred_model(data, my_copula, kappa, alpha, beta, phi)

    res_ours = run_fit_and_metrics_general(
        df=my_df,
        model_type="custom",
        fit_func_full=fit_func_full,
        fit_func_null=fit_func_null_no_phase,
        loglik_func=loglik_func
    )
    
    res = res_ours
    print(f"AIC_full={res['AIC_full']:.3f}, AIC_null={res['AIC_null']:.3f}, ΔAIC={res['Delta_AIC_null_to_full']:.3f}")

