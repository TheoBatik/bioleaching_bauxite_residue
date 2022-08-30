from gc import callbacks
from glob import glob
from src.functions import objective_function, X
from scipy.optimize import minimize, basinhopping
import numpy as np 
 

# # Inputs

# t_span = (c_prepared.iloc[0, 0], c_prepared.iloc[-1, 0])
# T = int(c_prepared.iloc[-1, 0])
# t_step = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0]
# t_end = 1
# time = c_prepared.iloc[:, 0].values
# state_0 = c_prepared.iloc[0, 1:].values
N_k = X.shape[1]
k_0 = np.ones( N_k )
k_bounds = [ (0.0000001, 2) for _ in range(N_k) ]


# Global minimization of objective function

class CustomBasinHop:
    def __init__(self, stepsize=0.1):
        self.stepsize = stepsize
        self.rng = np.random.default_rng()
    def __call__(self, k):
        s = self.stepsize
        step = [ self.rng.uniform(-1*s, 1*s) for _ in range(N_k) ]
        k += step
        return k
customer_hop = CustomBasinHop()

class CustomBounds:
    def __init__(self):
        self.k_upper = np.full(N_k, 1000)
        self.k_lower = np.zeros(N_k)
    def __call__(self, **kwargs):
        k = kwargs["x_new"]
        k_max = bool(np.all(k <= self.k_upper))
        k_min = bool(np.all(k >= self.k_lower))
        return k_max and k_min
custom_bounds = CustomBounds()

def print_min(x, f, accepted):
    print("at minimum %.1f accepted %d" % (f, int(accepted)))

minimizer_kwargs = {"method": "Nelder-Mead", "bounds": k_bounds}

global_minimizer = basinhopping(objective_function, k_0, minimizer_kwargs=minimizer_kwargs,
                niter=200, disp=True, take_step=customer_hop, accept_test=custom_bounds, callback=print_min)

k_global_min = global_minimizer.x 

print('k_global_min =', k_global_min)

q_global_min = global_minimizer.fun

print('q_global_min =', q_global_min)


# 429012771.5143282
# 190457022.25051206
# 429012771.5143282


# k_0 = np.array( [-1.15510211e+03 , 4.60248920e+02, -1.00290997e+00 , 3.68465225e+02 , -4.65093933e+01 , 1.19867505e-01] )

# local_minimizer = minimize(objective_function, k_0, method='Nelder-Mead', tol=1e-10, \
#     bounds=k_bounds, options={'disp': True}) # args=(), jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
# k_local_min = local_minimizer.x

# print('k_local_min =', k_local_min)

# q_local_min = local_minimizer.fun

# print('q_local_min =', q_local_min)


