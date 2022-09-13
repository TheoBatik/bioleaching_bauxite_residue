# Test the model and optimization algorithm
# On the following system of two species:
    # x1 -> x2 (k1)
    # x2 -> x1 (k2)

# Imports
import numpy as np
import matplotlib.pyplot as plt 
from src.model import evolve_network_for_test
from src.optimise import CustomHop, basinhopping, minimizer_kwargs


# Stoichiometry
A = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
B = np.flip( A, 0 )
Q = B - A


# Set initial conditions and the actual / target kinetics [ k = 2 ** k_exp ]
k_exp = np.array( [1, 1.2] )
x0 = np.array( [ 1, 0] )
print( '\nTarget kinetics, k = ', np.power( 2, k_exp) )


# Evolve network to generate simulation data (pseudo-measurements)
t_step = 0.01
N_t = 200
x_simulated = evolve_network_for_test(k_exp, x0, A, Q, t_step, N_t)
# print(x_simulated)


# Generate random initial estimate for k
low = 0
high = np.max(k_exp) * 2
k_estimate_exp = np.random.uniform(low=low, high=high, size=A.shape[0])
print( 'Initial estimate, k = ', np.power( 2, k_estimate_exp) )
print( 'Initial conditions = ', x0 )


# Objective function
def objective_for_test( k_exp ):
    # Predict network states given the kinetic parameters
    x_predicted = evolve_network_for_test( k_exp, x0, A, Q, t_step, N_t )
    # Compute the sum over time of the squared discrepency between the predicted and measured x
    error_squared =  0
    for t in range( N_t ):
        discrepency = x_predicted[ t, : ] - x_simulated[ t, : ]
        error_squared += discrepency.dot( discrepency )
    return error_squared


# Convergence tracker
k_exp_tracker = [ k_estimate_exp ]
objective_tracker = [ objective_for_test( k_estimate_exp ) ]
def track_convergence(k, f, accepted):
    if accepted and f < objective_tracker[-1]:
        print('\nBest estimate for k updated:')
        print('\tk = ', np.power( 2, k))
        # print('\tActual value of k = ', np.power( 2, k_exp) )
        print('\tObjective = ', f)
        objective_tracker.append(f)
        k_exp_tracker.append(k)


# Optimise kinetics - global minimization
print('\nStart basinhoppping...')
n_iterations = 2
display = False
stepsize = 0.5
custom_hop = CustomHop(stepsize)
N = 0
N_max = 1
while N < N_max:
    print('\nN = ', N)
    global_minimizer = basinhopping(objective_for_test, k_estimate_exp, minimizer_kwargs=minimizer_kwargs, T=500,
                        niter=n_iterations, disp=display, take_step=custom_hop, callback=track_convergence)
    k_estimate_exp = k_exp_tracker[-1]
    N += 1

# Model prediction
k_optimal = np.power( 2, k_exp_tracker[-1] )
x_predicted_best = evolve_network_for_test( k_optimal, x0, A, Q, t_step, N_t )


# Visualise - Actual vs Prediction
t_stop = t_step * N_t + t_step
time = np.arange(0, t_stop, t_step)
fig = plt.figure()
fig.suptitle( 'Predicted and Simulated States Over Time' )
plt.plot(time, x_predicted_best[:, 0], color='green', linestyle='--', linewidth=1, markersize=0 ) # marker='o', markersize=0.4 )
plt.plot(time, x_predicted_best[:, 1], color='blue', linestyle='--', linewidth=1, markersize=0 ) #marker='o', markersize=0.4 )
plt.plot(time, x_simulated[:, 0], color='green', linestyle='-', linewidth=1, markersize=0 )
plt.plot(time, x_simulated[:, 1], color='blue', linestyle='-', linewidth=1, markersize=0 )
plt.savefig('Actual vs Prediction.png')

print(k_optimal)


# Visualise - Convergence
fig, axs = plt.subplots( 2 )
# iterations = np.arange( 1, len( k_exp_tracker ) )
# axs[ 0 ].plot( iterations, k_exp_tracker[ : ], 'o' )
plt.savefig('Kinetic parameters over iterations.png')

# axs[0].set_title( 'Prediction' )
# axs[1].plot( x_simulated, time, '-' )
# axs[1].set_title( 'Actual' )