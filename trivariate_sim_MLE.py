import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from model.Jones_Pewsey import JonesPewsey, JonesPewseyPsi
from model.copulas import *
from numpy import cos, sin, tanh, log, exp, pi
from scipy.optimize import minimize
from scipy.special import lpmn, lpmv, gamma, gammaln
from tqdm import tqdm
from dataloader import nadalin
from dataloader import hilbert_phase, hilbert_envelope, bandpass_filter
from utils import scatter_unit_square, rank_uniform_2d

def sample_from_Nadalin():
    pac_mod = 1.0
    aac_mod = 0.0
    sim_method = "pink"
    rng = np.random.default_rng(12345)
    V1, Vlo, Vhi, t = nadalin(pac_mod, aac_mod, sim_method, rng)
    plt.plot(t,V1,label="V")
    plt.plot(t,Vlo, label="V_low")
    plt.plot(t,Vhi, label="V_high")
    plt.legend()
    plt.savefig("raw_signal")
    plt.clf()
    plt.plot(t[:100],V1[:100],label="V")
    plt.plot(t[:100],Vlo[:100], label="V_low")
    plt.plot(t[:100],Vhi[:100], label="V_high")
    plt.legend()
    plt.savefig("raw_signal_zoom")
    plt.close()
    return V1

def simulate_Nadalin(pac_mod = 1.0, aac_mod = 0.0, seed=0, has_plot=False):
    rng = np.random.default_rng(seed)
    sim_method = "pink"
    V1, Vlo, Vhi, t = nadalin(pac_mod, aac_mod, sim_method, rng)

    raw = V1
    data_lf = bandpass_filter(raw, low_cut=4, high_cut=7, fs=500)  # LF band-pass 
    data_hg = bandpass_filter(raw, low_cut=100, high_cut=140, fs=500)  # HG band-pass
    
    X1 = hilbert_envelope(data_lf)
    X2 = hilbert_envelope(data_hg)
    theta = hilbert_phase(data_lf)

    if has_plot:
        # --- 1枚目：raw + low + high を 1つの figure で ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # --- 上：raw signals ---
        ax1.plot(t, V1,  label="V")
        ax1.plot(t, Vlo, label="V_low")
        ax1.plot(t, Vhi, label="V_high")
        ax1.set_ylabel("raw signal")
        ax1.legend(loc="upper right")

        # --- 中：low freq ---
        ax2.plot(t, data_lf, label="low frequency bandpass", color="blue")
        ax2.plot(t, X1,      label="low amplitude (X1)",     color="red")
        ax2.set_ylabel("low band")
        ax2.legend(loc="upper right")

        # --- 下：high freq ---
        ax3.plot(t, data_hg, label="high frequency bandpass", color="blue")
        ax3.plot(t, X2,      label="high amplitude (X2)",     color="red")
        ax3.set_ylabel("high band")
        ax3.set_xlabel("t")
        ax3.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig("combined.png")
        plt.close(fig)

        # --- 2枚目：raw signal zoom ---
        plt.figure(figsize=(15, 6))
        plt.plot(t[:100], V1[:100],  label="V")
        plt.plot(t[:100], Vlo[:100], label="V_low")
        plt.plot(t[:100], Vhi[:100], label="V_high")
        plt.legend()
        plt.savefig("raw_signal_zoom.png")
        plt.close()


def simulate_CLL(
    n: int = 10000,
    copula: str = "gaussian",          # 'gaussian'|'clayton'|'gumbel'|'frank'|'amh'|'plackett'|'fgm'|'indep'
    cop_param = 0.6,                   # 定数 もしくは callable(theta)->float
    mu_true: float = 0.0,
    kappa_true: float = 0.8,
    alpha_true: float = 0.3,
    beta1_true: float = 0.1,
    beta2_true: float = 0.8,
    verbose: bool = False,
):
    """
    返り値:
        syn_data: shape (n,3) の ndarray [theta, x1, x2]
        true_params: (kappa_true, alpha_true, beta1_true, beta2_true)
    """
    true_params = (kappa_true, alpha_true, beta1_true, beta2_true)

    # --- Θ をサンプル（Jones-Pewsey がある想定。なければ von Mises にフォールバック）---
    try:
        GT = JonesPewsey(params=[mu_true, kappa_true, alpha_true])
        theta = GT.rvs(N=n)
    except NameError:
        theta = st.vonmises.rvs(kappa_true, loc=mu_true, size=n)

    # [-π, π) に正規化
    theta = (theta - mu_true + np.pi) % (2*np.pi) - np.pi

    # --- λ_i(θ) をベクトルで計算（数値安定化込み） ---
    rate1 = beta1_true * (1.0 - np.tanh(kappa_true) * np.cos(theta))
    rate2 = beta2_true * (1.0 - np.tanh(kappa_true) * np.cos(theta))
    rate1 = np.maximum(rate1, 1e-8)
    rate2 = np.maximum(rate2, 1e-8)

    # --- (U1,U2)|Θ をコピュラで生成（Rosenblatt: U1~Unif, U2 = h^{-1}(T|U1)）---
    rng = np.random.default_rng()
    U1 = clamp01(rng.random(n))
    T  = clamp01(rng.random(n))  # 条件付きCDFのターゲット

    # θ依存のパラメータにも対応
    if callable(cop_param):
        par = np.asarray([cop_param(th) for th in theta], float)
    else:
        par = np.full(n, float(cop_param), float)

    copula = copula.lower()
    U2 = np.empty(n, float)

    # family → hinv 関数のマップ（すべて signature: hinv(t, u, param)）
    hinv_map = {
        "gaussian": gaussian_hinv,
        "clayton" : clayton_hinv,
        "gumbel"  : gumbel_hinv,
        "frank"   : frank_hinv,
        "amh"     : amh_hinv,
        "plackett": plackett_hinv,
        "fgm"     : fgm_hinv,
        "indep"   : lambda t, u, _p: indep_hinv(t, u),  # param無視
    }
    if copula not in hinv_map:
        raise ValueError(f"unknown copula '{copula}'")

    hinv = hinv_map[copula]
    for j in range(n):
        U2[j] = hinv(T[j], U1[j], par[j])

    # --- ガンマ分位関数で戻す（scale=1/rate）---
    # U は端を避けて数値安定化
    u1_safe = np.clip(U1, 1e-12, 1-1e-12)
    u2_safe = np.clip(U2, 1e-12, 1-1e-12)
    x1 = st.gamma.ppf(u1_safe, a=alpha_true, loc=0.0, scale=1.0 / rate1)
    x2 = st.gamma.ppf(u2_safe, a=alpha_true, loc=0.0, scale=1.0 / rate2)

    syn_data = np.column_stack((theta, x1, x2))

    # ---------------- 可視化/検証ユーティリティ（任意） ----------------
    def rate1_of(th):
        return max(beta1_true * (1.0 - np.tanh(kappa_true) * np.cos(th)), 1e-8)
    def rate2_of(th):
        return max(beta2_true * (1.0 - np.tanh(kappa_true) * np.cos(th)), 1e-8)
    def pit1(x, th):
        return st.gamma.cdf(x, a=alpha_true, loc=0.0, scale=1.0 / rate1_of(th))
    def pit2(x, th):
        return st.gamma.cdf(x, a=alpha_true, loc=0.0, scale=1.0 / rate2_of(th))
    def ppf2(u, th):
        return st.gamma.ppf(u, a=alpha_true, loc=0.0, scale=1.0 / rate2_of(th))

    def conditional_pit_uv(data):
        arr = np.asarray(data, float)
        th_, x1_, x2_ = arr[:,0], arr[:,1], arr[:,2]
        u = np.array([pit1(x1_[i], th_[i]) for i in range(len(arr))], float)
        v = np.array([pit2(x2_[i], th_[i]) for i in range(len(arr))], float)
        return np.column_stack((theta, u, v))

    return syn_data, conditional_pit_uv(syn_data), true_params


kappa_hats = []
alpha_hats = []
beta1_hats = []
beta2_hats = []
phi_hats = []
error_count = 0
for _ in tqdm(range(1)):
    try:
        syn_data = simulate_Nadalin(); true_params = None
        # syn_data, con_syn_data, true_params = simulate_CLL(n=10000,copula="fgm",cop_param = 0.8)

        print("Data shape:", syn_data.shape)
        
        plt.figure(figsize=(5,5))
        plt.scatter(syn_data[:,0],syn_data[:,1], color="blue", label="(low freq phase vs low amplitude)")
        plt.scatter(syn_data[:,0],syn_data[:,2], color="orange", label="(low freq phase vs high amplitude)")
        plt.legend()
        plt.savefig("scatter_θ_vs_x.png")
        plt.clf()
        plt.figure(figsize=(5,5))
        plt.scatter(syn_data[:,1],syn_data[:,2], color="red")
        plt.savefig("scatter_x1_vs_x2.png")
        
        print(kendalltau(syn_data[:,1], syn_data[:,2], method="asymptotic")[0])
        # print(kendalltau(con_syn_data[:,1], con_syn_data[:,2], method="asymptotic")[0])
        # scatter_unit_square(rank_uniform_2d(syn_data[:,[1,2]]))
        # scatter_unit_square(rank_uniform_2d(con_syn_data[:,[1,2]]), filename="scatter_unit_square_conditional.png")
        
        # import pdb; pdb.set_trace()

        #### kappaとalphaをまずは推定.
        print("estimating first step ...")
        M = JonesPewsey()
        # M.score_matching(data=data[:,0])
        M.MLE(data = syn_data[:,0])
        _, kappa_hat, alpha_hat = M.params

        ###β1の推定
        def negative_log_likelihood_GM(theta, x, kappa, alpha, beta):
            P = lpmv(0,alpha-1,np.cosh(kappa))
            log_C = (
                np.log(2 * np.pi) +
                gammaln(alpha) +
                alpha * np.log(np.cosh(kappa)) +
                np.log(P) -
                alpha * np.log(beta)
            )

            if kappa <= 0:
                return np.inf
            if alpha <= 0:
                return np.inf
            if beta <= 0:
                return np.inf
            
            return -np.sum((alpha-1)*np.log(x) - beta * x * (1-np.tanh(kappa)*np.cos(theta)) - log_C)    

        def objective(beta, data):
            theta, x = data[:, 0], data[:, 1]
            kappa, alpha = kappa_hat, alpha_hat
            gm = negative_log_likelihood_GM(theta, x, kappa, alpha, beta)
            return gm


        print("estimating second step ...")
        result = minimize(lambda p: objective(p, syn_data[:,[0,1]]),
                        x0=np.array([0.5]) ,
                        method='L-BFGS-B',
                        bounds=[(1e-6, 10.0)] )
        beta1_hat = result.x.item()

        
        result = minimize(lambda p: objective(p, syn_data[:,[0,2]]),
                        x0=np.array([0.5]) ,
                        method='L-BFGS-B',
                        bounds=[(1e-6, 10.0)] )

        beta2_hat = result.x.item()
        print("="*30)
        print("周辺の真値")
        print(true_params)
        print("周辺の最尤推定")
        print(kappa_hat, alpha_hat, beta1_hat, beta2_hat)
        print("-"*30)



        def negative_log_likelihood_copula(theta, x1, x2, phi, kappa, alpha, beta1, beta2, eps=1e-12):
            """
            ベクトル theta, x1, x2 を受け取り、コピュラの負の対数尤度（合計）を返す。
            Xj|θ ~ Gamma(shape=alpha, rate=beta_j*(1 - tanh(kappa)*cos θ)) を仮定。
            """
            # 条件付きPIT（各サンプルの θ に対応するスカラー scale を使う）
            rate1 = np.maximum(beta1 * (1.0 - np.tanh(kappa) * np.cos(theta)), eps)
            rate2 = np.maximum(beta2 * (1.0 - np.tanh(kappa) * np.cos(theta)), eps)

            u = st.gamma.cdf(x1, a=alpha, loc=0.0, scale=1.0 / rate1)
            v = st.gamma.cdf(x2, a=alpha, loc=0.0, scale=1.0 / rate2)

            # 数値安定化（端での log(0) を避ける）
            u = np.clip(u, eps, 1.0 - eps)
            v = np.clip(v, eps, 1.0 - eps)

            c = frank_c(u, v, float(phi))
            c = np.clip(c, eps, None)  # φ≈±1 付近の極小を保護

            # “合計”の負の対数尤度（平均でもOK。最小化解は同じ）
            return -np.sum(np.log(c))
        
        def objective_copula(phi,data):
            theta, x1, x2 = data[:, 0], data[:, 1], data[:, 2]
            cop = negative_log_likelihood_copula(theta, x1, x2, phi, kappa_hat, alpha_hat, beta1_hat, beta2_hat)
            return cop

        print("estimating third step ...")
        result = minimize(lambda p: objective_copula(p, syn_data),
                        x0=np.array([0.5]) ,
                        method='L-BFGS-B',
                        # bounds=[(-0.999, 0.999)],
                        # options={"disp": True}
                        )
        phi_hat = result.x.item()
        print("-"*30)
        print("MLE of copula parameter φ")
        print(phi_hat)
        print("="*30)

        kappa_hats.append(kappa_hat)
        alpha_hats.append(alpha_hat)
        beta1_hats.append(beta1_hat)
        beta2_hats.append(beta2_hat)
        phi_hats.append(phi_hat)
    except:
        error_count += 1



### Post-procedure printing statistics.
from statistics import mean, variance, stdev
from math import sqrt

def se(arr):
    n = len(arr)
    if n <= 1:
        return float("nan")
    return stdev(arr) / sqrt(n)

def latex_row_pm(label, arr):
    m = mean(arr)
    s = se(arr)
    # 例: \kappa & 0.123 ± 0.01234 \\
    print(rf"{label} & ${m:.3f} \pm {s:.5f}$ \\")   # mathモードで \pm を出力

print("Error count: ", error_count)
print(r"\begin{tabular}{lc}")
print(r"\toprule")
print(r"Parameter & Mean $\pm$ SE \\")
print(r"\midrule")
latex_row_pm(r"$\kappa$",  kappa_hats)
latex_row_pm(r"$\alpha$",  alpha_hats)
latex_row_pm(r"$\beta_1$", beta1_hats)
latex_row_pm(r"$\beta_2$", beta2_hats)
latex_row_pm(r"$\phi$", phi_hats)
print(r"\bottomrule")
print(r"\end{tabular}")

