from src.input import A, X, states_measured, days
from scipy.optimize import minimize
import numpy as np

# PARAMS
t_step = days[1] - days[0] 
t_end = int(days[-1]) # int(states_measured.iloc[-1, 0])
state_0 = states_measured[0, :] # .iloc[0, 1:].values
N_k = A.shape[0]
# t_span = (states_measured.iloc[0, 0], states_measured.iloc[-1, 0])
# T = int(states_measured.iloc[-1, 0])
# k = np.ones(X.shape[1])**10**(-12)
# time = states_measured.iloc[:, 0].values

# Checks

# print('t_step = ', t_step )
# print('t_end = ', t_end )
# print('state_0 = ', state_0 )
# print('N_k = ', t_step )





# Evolve network
def evolve_network(kinetics):
    
    ''' Returns: predicted states'''
    
    # Copy initial conditions as current state
    state_current = state_0.copy()

    # Set initial conditions as first entry of output
    states_predicted = np.array(state_current) 

    # Loop through time
    for _ in range(t_end): # T

        # print('t =', t)
        # print('\t state_current (before) = \n', state_current)
        # print('\t kinetics (before) = \n', kinetics)
        # k = kinetics
        mass_action_vector = kinetics.copy()

        # print('\t state_current (before) = \n', state_current[0:5])
        
        # Construct the mass action vector
        # mass_action_vector = k
        # print('MAV (before)', mass_action_vector[0:5] )
        
        for r in range(N_k):
            # print('r =', r)
            # print(A[r,:])
            # print(np.prod( np.power(state_current, A[r,:]) ))
            mass_action_vector[r] *= np.prod( np.power(state_current, A[r,:]) )

        # print('\t MAV (after) = ', mass_action_vector[0:5])

        # Compute the concetration vector for the next step, c_next = c(t + 1)  
        c_next = state_current + t_step * X.dot(mass_action_vector)
       
        # Impose lower bound of zero on c_next
        min_ic = 0
        max_ic = max(c_next)
        c_next = np.clip(c_next, min_ic, max_ic)

        # print('\t c_next = ', c_next)

        # Stack c_next onto states_predicted 
        states_predicted = np.vstack((states_predicted, c_next))

        # Set c_next as the initial condition for the next step
        state_current = c_next
        
        # print('\t state_current = \n', state_current[0:5])
        # print('\t kinetics (after) = \n', kinetics)

        # print('\n')

    return states_predicted
 
# Objective funciton
def objective_function(kinetics):
    
    # Predict the network states given the kinetic parameters
    states_predicted = evolve_network(kinetics)

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

    return error_squared