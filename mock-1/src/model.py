from input import A, X, states_measured, days, hidden_states, c_header
import numpy as np
from scipy.integrate import solve_ivp

# USING solve_ivp

# System definitions
def reaction_network(t, state, *kinetics_exp):
    state_dot = X.dot( np.multiply( np.power( 2, kinetics_exp ), np.prod( np.power( np.clip(state, 0, None) , A[:,:] ), axis=1 )) )
    return state_dot

# Parameters
days_measured = days[:, 0]
t_span = ( days[0, 0], days[-1, 0] )
# print(' shape t, ', t_span)
state_0 = states_measured[0, :]
N_k = X.shape[ 1 ]
# lower = -2 
# upper = -1
# k_exp = np.around( np.random.uniform( low=lower, high=upper, size=N_k ), 2 )

# Results
# res = solve_ivp(reaction_network, t_span, state_0, args=k_exp, t_eval=days_measured, method='RK45')

# print("States predicted = \n", res.y)

def evolve_network( k_exp ):
    # print( '\nEvolve Network!\n')
    res = solve_ivp(reaction_network, t_span, state_0, args=k_exp, t_eval=days_measured, method='LSODA')
    return res.y


# Evolve network using finite difference method

# PARAMETERS
t_step = (days[1] - days[0]) #*10**(-3)
t_end = int(days[-1]) # int(states_measured.iloc[-1, 0])
state_0 = states_measured[0, :] # .iloc[0, 1:].values
N_k = A.shape[0]
# t_span = (states_measured.iloc[0, 0], states_measured.iloc[-1, 0])
# T = int(states_measured.iloc[-1, 0])
# k = np.ones(X.shape[1])**10**(-12)
# time = states_measured.iloc[:, 0].values

def evolve_network_using_finite_diff( k_exp ):
    
    ''' Returns: predicted states'''
    
    # Copy initial conditions as current state
    state_current = state_0.copy()

    # Set initial conditions as first entry of output
    states_predicted = np.array( state_current ) 

    # Loop through time
    for t in range( t_end ): 

        # print('t =', t)
        # print('\t state_current (before) = \n', state_current)
        # print('\t k_exp (before) = \n', k_exp)
        # k = k_exp
        # mass_action_vector = np.power( 2, k_exp.copy() ) 

        # print('\t state_current (before) = \n', state_current[0:5])
        
        # Construct the mass action vector
        # mass_action_vector = k
        # print('MAV (before)', mass_action_vector[0:5] )
        
        # for r in range(N_k):
            # print('r =', r)
            # print(A[r,:])
            # print(np.prod( np.power(state_current, A[r,:]) ))
        # print('\npower = ', np.power(state_current, A[:,:]) )
        # print('product = ', np.prod( np.power(state_current, A[:,:]), axis=1 ))
        mass_action_vector = np.multiply( np.power( 2, k_exp.copy() ), np.prod( np.power(state_current, A[:,:] ), axis=1 )) # Copy state_current to make it the same shape as A[:,:]
        # print( 'mass_action_vector = ', mass_action_vector )
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
        # print('\t k_exp (after) = \n', k_exp)

        # print('\n')

    return states_predicted


# Evolve network using finite difference method
def evolve_network_for_test( k_exp , x0, A, Q, t_step, t_end ):
    
    ''' Returns: predicted states'''

    # Copy initial conditions as current state
    state_current = x0.copy()

    # Set initial conditions as first entry of output
    states_predicted = np.array( state_current ) 

    # Loop through time
    for t in range( t_end ): 
        # Construct the mass action vector
        mass_action_vector = np.multiply( np.power( 2, k_exp.copy() ), np.prod( np.power(state_current, A[:,:] ), axis=1 )) # Copy state_current to make it the same shape as A[:,:]

        # Compute the concetration vector for the next step, c_next = c(t + 1)  
        c_next = state_current + t_step * Q.dot(mass_action_vector)
       
        # Impose lower bound of zero on c_next
        min_ic = 0
        max_ic = max(c_next)
        c_next = np.clip(c_next, min_ic, max_ic)

        # Stack c_next onto states_predicted 
        states_predicted = np.vstack((states_predicted, c_next))

        # Set c_next as the initial condition for the next step
        state_current = c_next

    return states_predicted