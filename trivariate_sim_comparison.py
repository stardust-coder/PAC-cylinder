import numpy as np
from scipy.special import gammaln
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import BSpline
from model.copulas import *
from model.GLMCFC import *
from model.CLL import MLE, log_likelihood_pred_model
from dataloader import nadalin
from dataloader import hilbert_phase, hilbert_envelope, bandpass_filter
import matplotlib.pyplot as plt
import seaborn as sns
########################################################################################################################
def count_params(obj):
    if np.isscalar(obj): return 1
    if isinstance(obj, (list, tuple, np.ndarray)):
        try:
            return int(np.prod(np.shape(obj))) or 1
        except Exception:
            return sum(count_params(v) for v in obj)
    if isinstance(obj, dict): return sum(count_params(v) for v in obj.values())
    return 1

########################################################################################################################

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
    import pdb; pdb.set_trace()

    if has_plot:
        # --- 1枚目：raw + low + high を 1つの figure で ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # --- 上：raw signals ---
        ax1.plot(t, V1,  label="V",color="blue")
        ax1.plot(t, Vhi, label="V_high",color="green")
        ax1.plot(t, Vlo, label="V_low",color="orange")
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

        # --- 1枚目：raw + low + high を 1つの figure で ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # --- 上：raw signals ---
        ax1.plot(t[:200], V1[:200],  label="V",color="blue")
        ax1.plot(t[:200], Vhi[:200], label="V_high",color="green")
        ax1.plot(t[:200], Vlo[:200], label="V_low",color="orange")
        ax1.set_ylabel("raw signal")
        ax1.legend(loc="upper right")

        # --- 中：low freq ---
        ax2.plot(t[:200], data_lf[:200], label="low frequency bandpass", color="blue")
        ax2.plot(t[:200], X1[:200],      label="low amplitude (X1)",     color="red")
        ax2.set_ylabel("low band")
        ax2.legend(loc="upper right")

        # --- 下：high freq ---
        ax3.plot(t[:200], data_hg[:200], label="high frequency bandpass", color="blue")
        ax3.plot(t[:200], X2[:200],      label="high amplitude (X2)",     color="red")
        ax3.set_ylabel("high band")
        ax3.set_xlabel("t")
        ax3.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig("combined_zoomed.png")
        plt.close(fig)

    
    mu_true = 0.0
    theta = (theta - mu_true + np.pi) % (2*np.pi) - np.pi     # mu=0前提に合わせる（ここではmu_true=0なので不要だが、一般にはこの1行でOK）

    syn_data = np.column_stack((theta, X1, X2))
    syn_data = syn_data[::10,:]


    return syn_data



def wrap_angle(theta):
    """角度を [-pi, pi] に正規化"""
    return (np.asarray(theta) + np.pi) % (2*np.pi) - np.pi

def make_periodic_knots(n_internal_knots, degree=3):
    # 周期Bスプライン用の等間隔ノット（位相用）
    # n_bases: 欲しい基底の本数
    # 位相は周期なので 0..2π で作り、入力は wrap して使う
    k = degree
    t_inner = np.linspace(-np.pi, np.pi, n_internal_knots - k + 1)
    # 周期拡張（端を k 個ずつ外側へ）
    dt = t_inner[1] - t_inner[0]
    t = np.r_[
        t_inner[0] - dt*np.arange(k,0,-1),
        t_inner,
        t_inner[-1] + dt*np.arange(1,k+1)
    ]
    return t  # len = n_internal_knots + k + 1

def bspline_basis(x, knots, degree=3):
    """
    Bスプライン基底行列を返す。
    x: 1D array
    knots: ノット列 (len = n_bases + degree + 1)
    """
    x = np.asarray(x)
    n_bases = len(knots) - (degree + 1)
    B = np.zeros((x.size, n_bases))
    for i in range(n_bases):
        c = np.zeros(n_bases); c[i] = 1.0
        B[:, i] = BSpline(knots, c, degree)(x)
    return B  # (N, n_bases)

# ------------- 設計行列 -------------
def design_matrix_Plow_Alow(phi_low, A_low, knots_phi, degree):
    """
    あなたの式：
      log μ = Σ_{k=1..n} β_k f_k(φ_low) + β_{n+1} A_low
              + β_{n+2} A_low sin(φ_low) + β_{n+3} A_low cos(φ_low)
    に対応する設計行列 X を作る。
    """
    phi = wrap_angle(phi_low)
    Bphi = bspline_basis(phi, knots_phi, degree)      # f_k(φ)
    inter = A_low.reshape(-1, 1)
    inter_sin = (A_low * np.sin(phi)).reshape(-1, 1)
    inter_cos = (A_low * np.cos(phi)).reshape(-1, 1)

    X = np.column_stack([Bphi, inter, inter_sin, inter_cos])
    return X  # 列順: [Bphi..., A_low, A_low*sin, A_low*cos]

def design_matrix_Alow(phi_low, A_low, knots_phi, degree):
    """
    ヌルモデル（位相依存なし）: log μ = β0 + β1 A_low
    ※ 公平性のため A_low の主効果は残す。
    """
    X = A_low.reshape(-1, 1)
    X = sm.add_constant(X)
    return X

def design_matrix_Plow(phi_low, A_low, knots_phi, degree):
    """
    φ_low モデル:
      A_high | φ_low ~ Gamma[μ, ν]
      log μ = Σ_{k=1..n} β_k f_k(φ_low)
    ->  φ の周期Bスプライン基底のみで設計行列を作る。
    """
    phi = wrap_angle(phi_low)
    Bphi = bspline_basis(phi, knots_phi, degree)   # shape: (n, n_basis)
    X = Bphi
    return X

def design_matrix_0(phi_low, A_low, knots_phi, degree):
    """
    Null: log μ = β0（定数のみ）
    """
    X = np.ones((len(A_low), 1))
    return X

# ------------- フィット -------------
def fit_gamma_glm(X, y):
    model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
    res = model.fit()
    return res

def run_fit_and_metrics_general(
    df,
    design_matrix_full=None,
    design_matrix_null=None,
    fit_func_full=None,
    fit_func_null=None,
    loglik_func=None,
    predict_func=None,
    model_type="gamma_glm",
    n_internal_knots=8,
    degree=3,
    n_A_grid=640,
    n_phi_grid=100,
    phi_min=-np.pi,
    phi_max=np.pi,
    quantile_range=(0.05, 0.95),
    custom_model_args=None,
):
    """
    汎用モデル評価関数：
    - Gamma GLM と自作 p(x2|x1,θ) の両方に対応。
    - custom モードでは full/null 両方のモデルを fit し、
      log-likelihood, AIC, ΔAIC を返す。
    """

    knots_phi = make_periodic_knots(n_internal_knots, degree)
    A_low = df["x1"].values
    phi_low = df["theta"].values
    y = df["x2"].values

    # =====================================================
    # ① Gamma GLM モード
    # =====================================================
    if model_type == "gamma_glm":
        X_full = design_matrix_full(phi_low, A_low, knots_phi, degree)
        X_null = design_matrix_null(phi_low, A_low, knots_phi, degree)

        res_full = fit_gamma_glm(X_full, y)
        res_null = fit_gamma_glm(X_null, y)
        aic_full, aic_null = res_full.aic, res_null.aic
        delta_aic = aic_null - aic_full

        # --- グリッド予測 ---
        qlo, qhi = np.quantile(A_low, quantile_range)
        A_grid = np.linspace(qlo, qhi, n_A_grid)
        phi_grid = np.linspace(phi_min, phi_max, n_phi_grid, endpoint=False)
        A_mat, phi_mat = np.meshgrid(A_grid, phi_grid, indexing="ij")
        A_vec, phi_vec = A_mat.ravel(), phi_mat.ravel()

        Xg_full = design_matrix_full(phi_vec, A_vec, knots_phi, degree)
        Xg_null = design_matrix_null(phi_vec, A_vec, knots_phi, degree)
        mu_full = np.exp(Xg_full @ res_full.params).reshape(n_A_grid, n_phi_grid)
        mu_null = np.exp(Xg_null @ res_null.params).reshape(n_A_grid, n_phi_grid)

        return dict(
            res_full=res_full, res_null=res_null,
            AIC_full=aic_full, AIC_null=aic_null,
            Delta_AIC_null_to_full=delta_aic,
            knots_phi=knots_phi,
            A_grid=A_grid, phi_grid=phi_grid,
            mu_full=mu_full, mu_null=mu_null
        )

    # =====================================================
    # ② Custom モード
    # =====================================================
    elif model_type == "custom":
        if (fit_func_full is None or fit_func_null is None or loglik_func is None):
            raise ValueError("custom モードでは fit_func_full/null と loglik_func を指定してください。")

        print("Estimating full model ...")
        params_full = fit_func_full(df, **(custom_model_args or {}))
        print("Estimating null model ...")
        params_null = fit_func_null(df, **(custom_model_args or {}))

        print("Computing log-likelihoods ...")
        LL_full = loglik_func(df, **params_full)
        LL_null = loglik_func(df, **params_null)

        k_full = 5 #count_params(params_full)
        k_null = 4 #count_params(params_null)
        AIC_full = 2*k_full - 2*LL_full
        AIC_null = 2*k_null - 2*LL_null
        delta_aic = AIC_null - AIC_full

        # --- 予測関数 (optional) ---
        mu_full = mu_null = None
        if predict_func is not None:
            qlo, qhi = np.quantile(df["x1"].values, quantile_range)
            A_grid = np.linspace(qlo, qhi, n_A_grid)
            phi_grid = np.linspace(phi_min, phi_max, n_phi_grid, endpoint=False)
            A_mat, phi_mat = np.meshgrid(A_grid, phi_grid, indexing="ij")
            A_vec, phi_vec = A_mat.ravel(), phi_mat.ravel()
            mu_full = predict_func(A_vec, phi_vec, params_full).reshape(n_A_grid, n_phi_grid)
            mu_null = predict_func(A_vec, phi_vec, params_null).reshape(n_A_grid, n_phi_grid)

        return dict(
            res_full=params_full,
            res_null=params_null,
            AIC_full=AIC_full,
            AIC_null=AIC_null,
            Delta_AIC_null_to_full=delta_aic,
            knots_phi=knots_phi,
            # A_grid=A_grid, phi_grid=phi_grid,
            mu_full=mu_full, mu_null=mu_null
        )

def plot_tail_dependence_by_theta(theta, X1, X2, nbins=6, q=0.95, tail='upper',
                                  q_low=None, fname=None):
    """
    theta, X1, X2: 同じ長さの ndarray
    nbins       : theta のビン数
    q           : 上側 tail の分位点（例: 0.95）
    tail        : 'upper' または 'lower'
    q_low       : 下側 tail 用の分位点（None なら 1-q を使用）
    fname       : 画像保存パス（None なら保存しない）
    """
    assert tail in ('upper', 'lower'), "tail must be 'upper' or 'lower'"
    if q_low is None:
        q_low = 1 - q  # 下側用のデフォルト閾値

    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    nrows = 2
    ncols = int(np.ceil(nbins / nrows))
    plt.figure(figsize=(6 * ncols, 5.5 * nrows))

    # 各ビンの推定 λ を格納（返り値）
    lambdas = []

    for k in range(nbins):
        mask = (theta >= edges[k]) & (theta < edges[k + 1])
        if not np.any(mask):
            lambdas.append(np.nan)
            continue

        x1b, x2b = X1[mask], X2[mask]

        # PIT（順位変換：ビン内ランクで U, V を作る）
        r1 = np.argsort(np.argsort(x1b))
        r2 = np.argsort(np.argsort(x2b))
        U = (r1 + 1) / (len(x1b) + 1)
        V = (r2 + 1) / (len(x2b) + 1)

        if tail == 'upper':
            thr = q
            tail_mask = (U > thr) & (V > thr)
            p_joint = np.mean(tail_mask)
            p_cond  = np.mean(U > thr)
            lam = p_joint / (p_cond + 1e-12)
            lam_label = r"λ_U"
        else:  # 'lower'
            thr = q_low
            tail_mask = (U < thr) & (V < thr)
            p_joint = np.mean(tail_mask)
            p_cond  = np.mean(U < thr)
            lam = p_joint / (p_cond + 1e-12)
            lam_label = r"λ_L"

        lambdas.append(lam)

        # プロット
        ax = plt.subplot(nrows, ncols, k + 1)
        ax.scatter(U, V, s=36, alpha=0.65)
        ax.scatter(U[tail_mask], V[tail_mask], s=36, alpha=0.99, color='r')
        if tail == 'upper':
            ax.axvline(thr, ls='--', color='gray'); ax.axhline(thr, ls='--', color='gray')
        else:
            ax.axvline(thr, ls='--', color='gray'); ax.axhline(thr, ls='--', color='gray')
        ax.set_title(f"θ ≈ {centers[k]:+.2f} rad\n{lam_label}≈{lam:.2f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("U"); ax.set_ylabel("V")

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.show()

    return np.array(lambdas)

if __name__ == "__main__":
    pac_mod, aac_mod= 1.0, 0.0
    syn_data = simulate_Nadalin(pac_mod=pac_mod, aac_mod=aac_mod) #theta, X1, X2
    print("Data shape:", syn_data.shape)
    my_df = pd.DataFrame(syn_data, columns=["theta", "x1", "x2"])
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
    plt.savefig(f"sample{pac_mod}-{aac_mod}-x1.png")

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
    plt.savefig(f"sample{pac_mod}-{aac_mod}-x2.png")

    theta, X1, X2 = syn_data[:,0], syn_data[:,1], syn_data[:,2]
    plot_tail_dependence_by_theta(theta, X1, X2, nbins=6, q=0.9, fname="tail_upper.png",tail='upper')
    plot_tail_dependence_by_theta(theta, X1, X2, nbins=6, q=0.9, fname="tail_lower.png",tail='lower')
    
    # my_copula = gumbel_c_phi_wrapper
    # my_copula = clayton_c
    my_copula = t_c_phi_wrapper


    def fit_func_full(df, **kwargs):
        data = df[["theta", "x1", "x2"]].values
        (params, phi_hat) = MLE(data, copula_func=my_copula)
        kappa, alpha, beta1, beta2 = params
        # phi_hat = [10.108885353117861,8.849005186242268,8.027959164683262,8.985379821591431,9.361487367496256,9.165217202040168]
        # phi_hat = [8.196156426204839,8.018286752449997,9.191966259744072,8.905208471331752,8.089576697051593,7.492467070514661]
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

    res_existing = run_fit_and_metrics_general(
        df=my_df,
        design_matrix_full=design_matrix_Plow_Alow,
        design_matrix_null=design_matrix_Alow,       
        model_type="gamma_glm",
        fit_func_full=None,
        fit_func_null=None,
        loglik_func=None
    )

    res_existing2 = run_fit_and_metrics_general(
        df=my_df,
        design_matrix_full=design_matrix_Plow_Alow,
        design_matrix_null=design_matrix_Plow,       
        model_type="gamma_glm",
        fit_func_full=None,
        fit_func_null=None,
        loglik_func=None
    )

    res_ours = run_fit_and_metrics_general(
        df=my_df,
        model_type="custom",
        fit_func_full=fit_func_full,
        fit_func_null=fit_func_null,
        loglik_func=loglik_func
    )

    res = res_existing
    print(f"AIC_full={res['AIC_full']:.3f}, AIC_null={res['AIC_null']:.3f}, ΔAIC={res['Delta_AIC_null_to_full']:.3f}")
    res = res_existing2
    print(f"AIC_full={res['AIC_full']:.3f}, AIC_null={res['AIC_null']:.3f}, ΔAIC={res['Delta_AIC_null_to_full']:.3f}")
    res = res_ours
    print(f"AIC_full={res['AIC_full']:.3f}, AIC_null={res['AIC_null']:.3f}, ΔAIC={res['Delta_AIC_null_to_full']:.3f}")

