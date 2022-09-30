from pickletools import optimize
from src.model import X, t_end, states_measured, evolve_network, hidden_states
from scipy.optimize import basinhopping, brute # minimize,
import numpy as np 


N_k = X.shape[ 1 ]
k_lower = np.zeros( N_k )
k_upper = np.full( N_k, 1000 )

# DEFINE THE OBJECTIVE FUNCTION

# Delete hidden states ( no data )
visible_states_measured = np.delete( states_measured, hidden_states, 1 )

# Normlize measured states
visible_states_measured_norm = visible_states_measured.copy()
n_columns = visible_states_measured_norm.shape[1]
column_maxes = []
for i in range( n_columns ):
    column = visible_states_measured_norm[:, i]
    column_max = column.max()
    column_norm = np.divide( column, column_max )
    # column_norm = (column - column.min()) / (column.max() - column.min())
    visible_states_measured_norm[:, i] = column_norm
    column_maxes.append( column_max )
column_maxes = np.array( column_maxes )
# print( 'column_maxes ', column_maxes )
# print( 'visible_states_measured_norm ', visible_states_measured_norm )


def objective( k_exp ):
    
    # Predict network states given the kinetic parameters
    states_predicted = evolve_network( k_exp )
    
    # print('\n states_predicted = ', states_predicted)
    # print('\n states_predicted shape  = ', states_predicted.shape)
    # print('\n states_measured = ', states_measured)
    # print('\n states_measured shape = ', states_measured.shape)

    # Delete hidden states and normalize
    visible_states_predicted = np.delete( states_predicted, hidden_states, 1 )
    visible_states_predicted_norm = visible_states_predicted / column_maxes
    # print( 'visible_states_predicted_norm = ', visible_states_predicted_norm)
    # Compute the sum over time of the squared discrepency between the predicted and measured states
    # print('t = ', t)
    # print('s_predicted_t', states_predicted[t, :]) 
    # print('s_measured_t', states_measured[t, :])
    discrepency = visible_states_predicted_norm[:, :] - visible_states_measured[:, :]
    error_squared = np.sum( np.multiply(discrepency, discrepency) )

    # Add a penalty for crossing the bounds
    k_too_low = bool(np.all(k_exp <= -2))
    if k_too_low:
        penalty = 10**9
        error_squared += penalty

    return error_squared





# SET OPTIMIZATION PARAMETERS

# t_span = (c_prepared.iloc[0, 0], c_prepared.iloc[-1, 0])
# T = int(c_prepared.iloc[-1, 0])
# t_step = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0]
# t_end = 1
# time = c_prepared.iloc[:, 0].values
# state_0 = c_prepared.iloc[0, 1:].values
lower = -2
upper = 1
k_exp_0 = np.around( np.random.uniform( low=lower, high=upper, size=N_k ), 2 ) #** 10 # 10 ** (-2)
# print('\nk_exp_0 = ', k_exp_0)

# k_bounds = [ ( 500, 1000) for _ in range(N_k) ]

k_exp_tracker = [ k_exp_0 ]
objective_tracker = [ objective( k_exp_0 ) ]
display = True

k_global_min = k_exp_0.copy()


# GLOBAL MINIMIZATION => basinhopping

class CustomHop:
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
    def __call__(self, k):
        s = self.stepsize
        # Adjust steps to account for k being exponentiated?
        high = np.log2( 2 - 2**(-s) )
        random_step = np.random.uniform(low=-s, high=high, size=k.shape)
        # print(' \nk_current = ', k)
        k += random_step
        # print(' k_new = ', k, '\n')
        return k

custom_hop = CustomHop()

def track_convergence(k, f, accepted):
    if f < objective_tracker[-1]: # and accepted 
        print('\t f = ', f)
        print('\t k_opt = ', k)
        objective_tracker.append(f)
        k_exp_tracker.append(k)

minimizer_kwargs = {} # {"bounds": k_bounds} # "method": "Nelder-Mead"} #

# N = 2
# # while N < 1:

# print('\nN = ', N, '\n')
print('\t k_global_min (start) = ', k_global_min)

# Minimize globally
global_minimizer = basinhopping(objective, k_global_min, minimizer_kwargs=minimizer_kwargs, # T=10472540367/4,
            niter=5, disp=False, take_step=custom_hop, callback=track_convergence)

# Update kinetics
k_global_min = k_exp_tracker[-1] # global_minimizer.x

print('\t objective_tracker[-1] = ', objective_tracker[-1])
print('\t global_minimizer.fun = ', global_minimizer.fun)
print('\t k_global_min (end) = ', k_global_min)

    # Update counter
    # N += 1


for i in range( len( objective_tracker) ):
    print('\ni = ', i)
    print('\t k_exp_tracker = ', np.around( k_exp_tracker[i] , 2) )
    k_converge = np.power( 2, k_exp_tracker )
    print('\t k_converge = ', k_converge[i]  )
    print('\t objective_tracker = ', objective_tracker[i])

print( '\nk_converge = ', k_converge)



states_predicted = evolve_network( k_global_min )
print( '\n ', 'Best prediction = ', states_predicted)





# BRUTE FORCE 

# k_lowest = 0.1
# k_upper = 2
# k_exp_lowest = np.log2(k_lowest)
# k_exp_upper = np.log2(k_upper)
# step = 0.1
# grid = ( slice( k_exp_lowest, k_exp_upper, step ), slice( k_exp_lowest, k_exp_upper, step ), slice( k_exp_lowest, k_exp_upper, step ) )
# brute_result = brute(objective, ranges=grid, args=(), Ns=None, full_output=0, finish=optimize.fmin, disp=display, workers=1)

# print( 'Global min = ', brute_result[0] ) # global minimum
# print( 'Function Value = ', brute_result[1] ) # function value at global minimum


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


