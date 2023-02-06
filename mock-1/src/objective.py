from pickletools import optimize
from model import X, t_end, states_measured, evolve_network, evolve_network_using_finite_diff, hidden_states, t_step, c_header
from scipy.optimize import basinhopping, brute # minimize,
import numpy as np
import matplotlib.pyplot as plt




# DEFINE THE OBJECTIVE FUNCTION

# Delete hidden states ( no measured data ) and normalize using max of each column
visible_states_measured = np.delete( states_measured, hidden_states, 1 )
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
# print( 'visible_states_measured ', visible_states_measured )

# Objective function using finite differences to evolve network
def objective_using_finite_diff( k_exp ):
    
    # Predict network states given the kinetic parameters
    states_predicted = evolve_network_using_finite_diff( k_exp )
    
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


# Objective function using scipy solve_ivp to evolve network


def objective( k_exp ):
    
    # Predict network states given the kinetic parameters
    states_predicted = evolve_network( k_exp )
    
    # print('\n states_predicted = ', states_predicted)
    # print('\n states_predicted shape  = ', states_predicted.shape)

    # print('\n visible_states_measured = ', visible_states_measured)
    # print('\n visible_states_measured shape = ', visible_states_measured.shape)

    # Delete hidden states and normalize
    visible_states_predicted = np.delete( states_predicted, hidden_states, 0 )

    # print('\n visible_states_predicted = ', visible_states_predicted)
    # print('\n visible_states_predicted shape  = ', visible_states_predicted.shape)

    visible_states_predicted_norm = visible_states_predicted.T / column_maxes
    # print( 'visible_states_predicted_norm = ', visible_states_predicted_norm)
    # Compute the sum over time of the squared discrepency between the predicted and measured states
    # print('t = ', t)
    # print('s_predicted_t', states_predicted[t, :]) 
    # print('s_measured_t', states_measured[t, :])
    discrepency = visible_states_predicted_norm[:, :] - visible_states_measured_norm[:, :]
    # print('\n discrepency', discrepency)
    error_squared = np.sum( np.multiply( discrepency, discrepency ) )

    # Add a penalty for crossing the bounds
    # k_too_low = bool(np.all(k_exp <= -2))
    # if k_too_low:
    #     penalty = 10**12
    #     error_squared += penalty

    return error_squared


# SET OPTIMIZATION PARAMETERS

# t_span = (c_prepared.iloc[0, 0], c_prepared.iloc[-1, 0])
# T = int(c_prepared.iloc[-1, 0])
# t_step = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0]
# t_end = 1
# time = c_prepared.iloc[:, 0].values
# state_0 = c_prepared.iloc[0, 1:].values
N_k = X.shape[ 1 ]
lower_bound = -2
upper_bound = 2
k_exp_0 = np.around( np.random.uniform( low=lower_bound, high=upper_bound, size=N_k ), 2 ) #** 10 # 10 ** (-2)
# print('\nk_exp_0 = ', k_exp_0)

# k_bounds = [ ( 500, 1000) for _ in range(N_k) ]

k_exp_tracker = [ k_exp_0 ]
objective_tracker = [ objective( k_exp_0 ) ]
display = True

k_global_min = k_exp_0.copy()


# GLOBAL MINIMIZATION => basinhopping

class CustomHop:
    def __init__(self, stepsize=1):
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
        # print('\t f = ', f)
        # print('\t k_opt = ', k)
        objective_tracker.append(f)
        k_exp_tracker.append(k)

minimizer_kwargs = {"method": "L-BFGS-B"} #{"method": "Nelder-Mead"} # {"bounds": k_bounds} # } #

T = 0 # temperature
N = 0
while N < 1:

    # print('\nN = ', N, '\n')
    # print('\t k_global_min (latest) = ', k_global_min)   
    
    # Minimize globally
    global_minimizer = basinhopping(objective, k_global_min, minimizer_kwargs=minimizer_kwargs, T=T,
                niter=2, disp=display, take_step=custom_hop, callback=track_convergence)

    # Update kinetics and temperature
    delta_obj = objective_tracker[-2] - objective_tracker[-1]
    # print( '\t delta_obj', delta_obj)
    T = delta_obj
    # print('\t T', T)

    if delta_obj < 10e-5:
        T = 1/delta_obj
        k_global_min = np.around( np.random.uniform( low=lower_bound, high=upper_bound, size=N_k ), 2 )
    else:
        k_global_min = k_exp_tracker[-1] # global_minimizer.x

    N += 1

print('\t objective_tracker[-1] = ', objective_tracker[-1])
print('\t global_minimizer.fun = ', global_minimizer.fun)
print('\t k_global_min (end) = ', k_global_min)

k_converge = np.power( 2, k_exp_tracker )
print( '\nk_converge = ', k_converge)

for i in range( len( objective_tracker ) ):
    print('\ni = ', i)
    print('\t k_exp_tracker = ', np.around( k_exp_tracker[i] , 2) )
    print('\t k_converge = ', k_converge[i]  )
    print('\t objective_tracker = ', objective_tracker[i])



states_predicted_best = evolve_network( k_global_min )
print( '\n ', 'Best prediction = ', states_predicted_best)



# Visualise - Actual vs Predicted states

N_t = len( states_measured[:, 0] )
t_stop = t_step * N_t # + t_step
time = np.arange(0, t_stop, t_step)

# Citric acid and aqueous Sc
# fig = plt.figure()
# fig.suptitle( 'Predicted States and Measured States Over Time' )
# plt.plot(time, states_predicted_best[0, :], color='green', linestyle='--', linewidth=1, markersize=0 ) # marker='o', markersize=0.4 )
# plt.plot(time, states_predicted_best[-2, :], color='blue', linestyle='--', linewidth=1, markersize=0 ) #marker='o', markersize=0.4 )
# plt.plot(time, states_measured[:, 0], color='green', linestyle='-', linewidth=1, markersize=0 )
# plt.plot(time, states_measured[:, -2], color='blue', linestyle='-', linewidth=1, markersize=0 )
# fig.tight_layout()
# plt.savefig( 'Subsystem - Predicted States and Measured States Over Time.png' )

# All species
N_c = len( states_measured[0, :] )
fig, axs = plt.subplots( 3, 2 )

for i in range(3):

    axs[i, 0].plot( time, states_predicted_best[i, :], color='green', linestyle='--', linewidth=1, markersize=0 )
    axs[i, 0].plot( time, states_measured[:, i], color='blue', linestyle='-', linewidth=1, markersize=0 )

    axs[i, 1].plot( time, states_predicted_best[i + 3, :], color='green', linestyle='--', linewidth=1, markersize=0 )
    axs[i, 1].plot( time, states_measured[:, i + 3], color='blue', linestyle='-', linewidth=1, markersize=0 )

# axs[1].plot( time, states_predicted_best[-2, :], 'o', linestyle='--', linewidth=1 )
fig.tight_layout()
plt.savefig( 'Subsystem - Predicted States and Measured States Over Time (all species).png' )

# Citric vs aqueous SC
fig, axs = plt.subplots( 2 )
axs[0].plot( time, states_predicted_best[0, :], color='green', linestyle='--', linewidth=1, markersize=0 )
axs[0].plot( time, states_measured[:, 0], color='blue', linestyle='-', linewidth=1, markersize=0 )
axs[0].set_title( str( c_header[0] ) )
axs[1].plot( time, states_predicted_best[-2, :], color='green', linestyle='--', linewidth=1, markersize=0 )
axs[1].plot( time, states_measured[:, -2], color='blue', linestyle='-', linewidth=1, markersize=0 )
axs[1].set_title( str( c_header[-2] ) )
fig.tight_layout()
plt.savefig( 'Subsystem - Citric acid vs aqueous Sc over time.png' )



# Visualise - Convergence

# Kinetics
k_converging = np.power( 2, k_exp_tracker )
iterations = np.arange( 0, len( k_exp_tracker ) )
fig, axs = plt.subplots( 3, 2 )
for i in range(3):
    axs[ i, 0 ].plot( iterations, k_converging[ : , i], 'o', linestyle='--', linewidth=1, markersize=0.1 )
    axs[ i, 0 ].set_title( f'Kinetic param. {i}' )
    axs[ i, 1 ].plot( iterations, k_converging[ : , i + 3], 'o', linestyle='--', linewidth=1, markersize=0.1 )
    axs[ i, 1 ].set_title( f'Kinetic param. {i + 3}' )
fig.tight_layout()
plt.savefig('Subsystem - kinetic parameters by iteration number.png')

# Objective function
fig = plt.figure()
plt.plot(iterations, objective_tracker, 'o', linestyle='--', linewidth=1, markersize=0.1)
plt.savefig('Subsystem - objective function value by iteration number.png')
