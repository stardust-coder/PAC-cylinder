import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmn, lpmv, gamma, gammaln
from math import log, tanh, cos, cosh, pi, exp

def MLE(data):
    '''
    data : numpy array [[theta,x],[theta,x], ...]
    '''
    mu = np.angle(np.mean(np.exp(1j * data[:,0])))
    data = data.tolist()

    def negative_log_likelihood(params):
        '''
        Notes: For stable calculation of MLE, kappa should be smaller than 0.5 because of lpmv function.
        '''
        kappa, alpha, beta = params
        P = lpmv(0,alpha-1,np.cosh(kappa))
        # C = 2 * pi * gamma(alpha) * (np.cosh(kappa)**alpha) * P
        # C = C / (beta**alpha)
        # print(beta, alpha, C)
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
        log_L = 0
        for (theta,x) in data:
            inc = (alpha-1)*np.log(x) - beta * x * (1-np.tanh(kappa)*np.cos(theta-mu)) - log_C
            log_L += inc
        return -log_L
    
    # 初期値
    initial_params = [1,1,1]

    # 最適化
    result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None),(1e-6, None),])
    
    theta_hat = result.x
    hess_inv = result.hess_inv.todense()  # L-BFGS-B ではこれで取得できる
    se = np.sqrt(np.diag(hess_inv))  # 標準誤差 = 分散の平方根
    ci_lower = theta_hat - 1.96 * se
    ci_upper = theta_hat + 1.96 * se

    # 結果の表示
    kappa_mle, alpha_mle, beta_mle = result.x
    print(f"最尤推定: mu:{mu:.4f},kappa:{kappa_mle:.4f},alpha:{alpha_mle:.4f},beta:{beta_mle:.4f}")
    return (mu, kappa_mle, alpha_mle, beta_mle), ci_lower, ci_upper

import numpy as np
from scipy.special import lpmv, gamma

def GM(data, mu, kappa, alpha, beta):
    """
    data: shape (N, 2) の numpy 配列。各行が [theta, l]
    mu, kappa, alpha, beta: モデルパラメータ（スカラー）
    """
    theta = data[:, 0]
    l = data[:, 1]

    # 定数項の計算
    P = lpmv(0, alpha - 1, np.cosh(kappa))  # P_{alpha-1}^0(cosh(kappa))
    C = 2 * np.pi * gamma(alpha) * (np.cosh(kappa) ** alpha) * P
    C = C / (beta ** alpha)

    # [-π, π) に正規化された角度差
    delta = np.mod(theta - mu + np.pi, 2 * np.pi) - np.pi

    # 尤度の計算
    exponent = -beta * l * (1 - np.tanh(kappa) * np.cos(delta))
    density = (1 / C) * (l ** (alpha - 1)) * np.exp(exponent)

    return density

def GM_C(data,mu,kappa,alpha,beta):
    theta = data[:, 0]
    l = data[:, 1]
    P = lpmv(0, alpha - 1, np.cosh(kappa))
    res = (1 - np.tanh(kappa) * np.cos(theta-mu)) ** (-alpha)
    res /= 2 * np.pi * (np.cosh(kappa) ** alpha) * P
    return res

def GM_L(data,mu,kappa,alpha,beta):
    from scipy.special import i0
    theta = data[:, 0]
    l = data[:, 1]

    # 定数項の計算
    P = lpmv(0, alpha - 1, np.cosh(kappa))  # P_{alpha-1}^0(cosh(kappa))
    C = 2 * np.pi * gamma(alpha) * (np.cosh(kappa) ** alpha) * P
    C = C / (beta ** alpha)
    return (1/C) * 2 * np.pi * (l**(alpha-1)) * np.exp(-beta*l) * i0(beta * l * np.tanh(kappa))

def MI(data,params):
    mu,kappa,alpha,beta = params
    f_GM = GM(data,mu,kappa,alpha,beta)
    f_GM_C = GM_C(data,mu,kappa,alpha,beta)
    f_GM_L = GM_L(data,mu,kappa,alpha,beta)
    
    mi = np.mean(np.log(f_GM) - np.log(f_GM_C) - np.log(f_GM_L))
    print("Calculated Mutual Information = ", mi)
    return mi

def sample_from_GM(N, params):
    mu,kappa,alpha,beta = params

    def target_dist(theta):
        # return np.exp(-0.5 * x**2)
        return (1 - tanh(kappa) * cos(theta - mu))**(-alpha)

    # 提案分布（サンプルとPDF）
    def proposal_sample(size):
        return np.random.uniform(-pi, pi, size)

    def proposal_pdf(x):
        return 1 / (2 * pi)  # 一様分布 の密度

    # 定数M（target / proposalの上限値を見積もる）
    M = ((1 - tanh(kappa))**(-alpha)) / (1 / (2 * pi) )  # ターゲット関数が最大になるのは x = 0

    # リジェクションサンプリング本体
    def rejection_sampling(n_samples):
        samples = []
        while len(samples) < n_samples:
            x = proposal_sample(1)[0]
            u = np.random.uniform(0, 1)
            if u < target_dist(x) / (M * proposal_pdf(x)):
                samples.append(x)
        return np.array(samples)

    # サンプル生成
    samples = rejection_sampling(N)

    import scipy
    pair_samples = []
    for theta in samples.tolist():
        r = scipy.stats.gamma(a=alpha,scale=1/(beta*(1-tanh(kappa)*cos(theta-mu)))).rvs(size=1)
        pair_samples.append([theta, r.item()])
    
    return np.array(pair_samples)