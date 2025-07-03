import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
from scipy.special import gamma, loggamma
from utils import visualize, density_plot, joint_plot

# 再度必要な依存コードを復元
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gamma, loggamma

def polar_density(data, params):
    theta, x = data[:, 0], data[:, 1]
    mu, kappa, alpha, beta = params
    c = 1 - np.tanh(kappa) * np.cos(theta - mu)
    return x ** (alpha - 1) * np.exp(-beta * x * c)

def log_normalizing_constant(mu, kappa, alpha, beta):
    def integrand(theta):
        denom = 1 - np.tanh(kappa) * np.cos(theta - mu)
        return denom ** (-alpha)

    integral, _ = quad(integrand, 0, 2 * np.pi, limit=100)
    log_z = loggamma(alpha) - alpha * np.log(beta) + np.log(integral)
    return log_z

def Q_function(params, data, gamma_k):
    mu, kappa, alpha, beta = params
    theta, x = data[:, 0], data[:, 1]
    c = 1 - np.tanh(kappa) * np.cos(theta - mu)
    log_unnorm = (alpha - 1) * np.log(x) - beta * x * c
    log_f = log_unnorm - log_normalizing_constant(mu, kappa, alpha, beta)
    return np.sum(gamma_k * log_f)

# 評価関数を再掲
def evaluate_fit(model, data):
    n = len(data)
    k = model.n_components
    n_params = 4 * k + (k - 1)

    ll = model.log_likelihood(data)
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n) * n_params

    theta_grid = np.linspace(0, 2 * np.pi, 200)
    x_grid = np.linspace(0.01, np.max(data[:, 1]), 200)
    TH, XX = np.meshgrid(theta_grid, x_grid)
    points = np.column_stack([TH.ravel(), XX.ravel()])

    # 各成分ごとの密度
    component_densities = []
    for k in range(model.n_components):
        comp_dens = model.weights[k] * model.density(points, model.params[k])
        component_densities.append(comp_dens.reshape(200, 200))

    # 可視化
    plt.figure(figsize=(12, 10))
    for k, comp in enumerate(component_densities):
        plt.contour(TH, XX, comp, levels=5, linewidths=1.5, alpha=0.7, label=f"Component {k+1}")

    plt.scatter(data[:, 0], data[:, 1], s=5, color='black', alpha=0.3, label='Observed')
    plt.xlabel("θ (radian)")
    plt.ylabel("x")
    plt.title("Component-wise Model Fit vs Observed Data")
    plt.legend()
    plt.colorbar(label="Density (per component)")
    plt.show()
    plt.savefig("output/evaluate_fit.png")

    return {
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic
    }

class EMAlgorithm:
    def __init__(self, params_init, n_components, weights_init=None):
        self.params = params_init
        self.n_components = n_components
        if weights_init is not None:
            assert len(weights_init) == n_components
            self.weights = np.array(weights_init)
        else:
            self.weights = np.ones(n_components) / n_components

    def density(self, data, params):
        mu, kappa, alpha, beta = params
        unnorm = polar_density(data, params)
        log_z = log_normalizing_constant(mu, kappa, alpha, beta)
        return unnorm / np.exp(log_z)

    def e_step(self, data):
        n = len(data)
        gamma = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            gamma[:, k] = self.weights[k] * self.density(data, self.params[k])
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def m_step(self, data, gamma):
        self.weights = np.sum(gamma, axis=0) / len(data)
        for k in range(self.n_components):
            def neg_Q(p):
                return -Q_function(p, data, gamma[:, k])

            mu0, kappa0, alpha0, beta0 = self.params[k]
            res = minimize(
                neg_Q,
                x0=np.array([mu0, kappa0, alpha0, beta0]),
                bounds=[(0, 2 * np.pi), (1e-3, 10), (1e-3, 10), (1e-3, 10)],
                method='L-BFGS-B'
            )
            if res.success:
                self.params[k] = tuple(res.x)
            else:
                print(f"Warning: optimization failed for component {k}")

    def log_likelihood(self, data):
        ll = 0.0
        for i in range(len(data)):
            prob = sum(self.weights[k] * self.density(data[i:i+1], self.params[k])[0]
                       for k in range(self.n_components))
            ll += np.log(prob)
        return ll

    def fit(self, data, tol=1e-4, max_iter=10000):
        prev_ll = -np.inf
        for step in range(max_iter):
            gamma = self.e_step(data)
            self.m_step(data, gamma)
            ll = self.log_likelihood(data)
            print(f"Iter {step+1}, Log-likelihood: {ll:.4f}")
            tmp = [list(x) for x in self.params]
            for i in range(len(tmp)):
                print(self.weights[i],[x.item() for x in tmp[i]])
            if np.abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        


def sample_wCvM(N,mu,kappa,alpha,beta):
    from scipy.stats import wrapcauchy
    from scipy.stats import weibull_min
    # パラメータ: location μ ∈ [0, 2π), concentration ρ ∈ (0, 1)
    # scipyでは scale=1固定で loc=μ, c=ρ を指定
    theta_samples = wrapcauchy.rvs(c=np.tanh(kappa/2), loc=mu, size=N)

    samples = []
    for theta in theta_samples:    
        theta = np.mod(theta, 2 * np.pi)
        # パラメータ: shape = α, scale = β
        # scipy: weibull_min(c=shape, scale=scale)
        x_sample = weibull_min.rvs(c=alpha, scale=beta*((1-np.tanh(kappa)*np.cos(theta-mu))**(1/alpha)), size=1)
        samples.append([theta.item(),x_sample.item()])
    return np.array(samples)


if __name__ == "__main__":
    print("Hello world.")

    data_ls = []
    for _ in range(1000):
        if np.random.random() < 0.3:
            data_ls.append(sample_wCvM(N=1, mu=0,kappa=1,alpha=2,beta=1))
        else:
            data_ls.append(sample_wCvM(N=1, mu=3,kappa=0.5,alpha=2,beta=0.5))
    data = np.vstack(data_ls)
    joint_plot(data[:,0],data[:,1])

    # import pdb; pdb.set_trace()

    params_init = [
        (0.0, 1.0, 2.0, 1.0),         # 混合成分1
        (3.0, 2.0, 3.0, 0.5)        # 混合成分2
    ]
    model = EMAlgorithm(params_init,len(params_init))
    model.fit(data)
    #Fit具合を可視化
    print(evaluate_fit(model,data))