from src.model import X, evolve_network_for_test, t_end, states_measured, evolve_network
from scipy.optimize import minimize, basinhopping
import numpy as np 


N_k = X.shape[ 1 ]
k_lower = np.zeros( N_k )
k_upper = np.full( N_k, 1000 )

# DEFINE THE OBJECTIVE FUNCTION

def objective( k_exp ):
    
    # Predict network states given the kinetic parameters
    states_predicted = evolve_network( k_exp )

    # print('\n states_predicted = ', states_predicted)
    # print('\n states_predicted shape  = ', states_predicted.shape)
    # print('\n states_measured = ', states_measured)
    # print('\n states_measured shape = ', states_measured.shape)

    # Compute the sum over time of the squared discrepency between the predicted and measured states
    error_squared =  0
    for t in range(t_end):
        # print('t = ', t)
        # print('s_predicted_t', states_predicted[t, :]) 
        # print('s_measured_t', states_measured[t, :])
        discrepency = states_predicted[t, :] - states_measured[t, :]
        error_squared += discrepency.dot(discrepency)

    # # Add a penalty for crossing the bounds
    # k_max_bool = bool(np.all(k_exp <= k_upper))
    # k_min_bool = bool(np.all(k_exp >= k_lower))
    # if k_max_bool and k_min_bool:
    #     penalty = 10**8

    return error_squared # + penalty





# SET OPTIMIZATION PARAMETERS

# t_span = (c_prepared.iloc[0, 0], c_prepared.iloc[-1, 0])
# T = int(c_prepared.iloc[-1, 0])
# t_step = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0]
# t_end = 1
# time = c_prepared.iloc[:, 0].values
# state_0 = c_prepared.iloc[0, 1:].values
lower = 5
upper = 6
k_exp_0 = np.around( np.random.uniform(low=lower, high=upper, size=N_k) , 2 ) #** 10 # 10 ** (-2)
# print('\nk_exp_0 = ', k_exp_0)

# k_bounds = [ ( 500, 1000) for _ in range(N_k) ]

k_exp_tracker = [ k_exp_0 ]
objective_tracker = [ objective( k_exp_0 ) ]
display = True

k_global_min = k_exp_0.copy()


# GLOBAL MINIMIZATION => basinhopping

class CustomHop:
    def __init__(self, stepsize=3):
        self.stepsize = stepsize
    def __call__(self, k):
        s = self.stepsize
        # Adjust steps to account for k being exponentiated?
        random_step = np.random.uniform(low=-s, high=s/2, size=k.shape)
        # print(' \nk_current = ', k)
        k += random_step
        # print(' k_new = ', k, '\n')
        return k

custom_hop = CustomHop()

def track_convergence(k, f, accepted):
    # if accepted:
    #     print('Accepted: k is within bounds')
    # else: 
    #     print('Not Accepted: k is out of bounds')
    if accepted and f < objective_tracker[-1]: # and accepted 
        # print('\t f = ', f)
        # print('\t k_opt = ', k)
        objective_tracker.append(f)
        k_exp_tracker.append(k)

minimizer_kwargs = {} # {"bounds": k_bounds} # "method": "Nelder-Mead"} #


if __name__ == "__main__":


    N = 0


    while N < 8:

        print('\nN = ', N, '\n')
        print('\t k_global_min (start) = ', k_global_min)
        
        # Minimize globally
        global_minimizer = basinhopping(objective, k_global_min, minimizer_kwargs=minimizer_kwargs, # T=10,
                    niter=10, disp=display, take_step=custom_hop, callback=track_convergence)

        # Update kinetics
        k_global_min = global_minimizer.x # k_exp_tracker[-1]

        # print('\t objective_tracker[-1] = ', objective_tracker[-1])
        # print('\t global_minimizer.fun = ', global_minimizer.fun)
        # print('\t k_global_min (end) = ', k_global_min)

        # Update counter
        N += 1


    for i in range( len( objective_tracker) ):
        print('\ni = ', i)
        print('\t k_exp_tracker = ', np.around( k_exp_tracker[i] , 2) )
        k_converge = np.power( 2, k_exp_tracker )
        print('\t k_converge = ', k_converge[i]  )
        print('\t objective_tracker = ', objective_tracker[i])




    # print('k_global_min =', k_global_min)

    # q_global_min = global_minimizer.fun

    # print('q_global_min =', q_global_min)


    # 429012771.5143282
    # 190457022.25051206
    # 429012771.5143282
    # 9758208839.916275

    # k_exp_0 = np.array( [-1.15510211e+03 , 4.60248920e+02, -1.00290997e+00 , 3.68465225e+02 , -4.65093933e+01 , 1.19867505e-01] )



    # local_minimizer = minimize(objective, k_exp_0, method='Nelder-Mead', tol=1e-10, \
    #     bounds=k_bounds, options={'disp': True}) # args=(), jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    # k_local_min = local_minimizer.x

    # print('k_local_min =', k_local_min)

    # q_local_min = local_minimizer.fun

    # print('q_local_min =', q_local_min)


