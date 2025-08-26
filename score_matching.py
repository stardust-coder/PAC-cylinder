from model.Gam_vM import Gamma_von_Mises
import numpy as np
from time import time
from statistics import mean, stdev, variance

est_k = []
est_a = []
est_b = []

comp = []
for _ in range(30):
    simulate = Gamma_von_Mises([0,1.0,1.0,0.2])
    data = simulate.rvs(1000)
    # print(data)

    model = Gamma_von_Mises()
    start_time = time()
    model.score_matching(data, eta=1e-3)
    # model.MLE(data)
    comp_time = time()-start_time
    est_k.append(model.params[1].item())
    est_a.append(model.params[2].item())
    est_b.append(model.params[3].item())

    comp.append(comp_time)

print("[Estimator (κ）]","mean:", mean(est_k), "var:", variance(est_k))
print("[Estimator (α）]","mean:", mean(est_a), "var:", variance(est_a))
print("[Estimator (β）]","mean:", mean(est_b), "var:", variance(est_b))

print("[Time]","mean:", mean(comp), "var:", variance(comp))



