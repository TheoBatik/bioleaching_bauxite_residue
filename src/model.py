from src.input import A, X, states_measured, days
import numpy as np

# PARAMS
t_step = (days[1] - days[0])*10**(-3)
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





# Evolve network using finite difference method
def evolve_network( k_exp ):
    
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
        mass_action_vector = np.power( 2, k_exp.copy() ) 

        # print('\t state_current (before) = \n', state_current[0:5])
        
        # Construct the mass action vector
        # mass_action_vector = k
        # print('MAV (before)', mass_action_vector[0:5] )
        
        for r in range(N_k):
            # print('r =', r)
            # print(A[r,:])
            # print(np.prod( np.power(state_current, A[r,:]) ))
            mass_action_vector[r] *= np.prod( np.power(state_current, A[r,:]) ) # Copy state_current to make it the same shape as A[:,:]

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






# def check_input_dimensions(number_species, number_reactions, stoich_X, stoich_A):
#     if not (number_species == stoich_X.shape[0] and number_species == stoich_A.shape[1]):
#         raise('The dimension of the initial conditions must match that of the stoichiometric matrices.')
#     if not (number_reactions == stoich_X.shape[1] and number_reactions == stoich_A.shape[0]):
#         raise('The dimension of the kinetics vector must match that of the stoichiometric matrices.')


# # Inputs
# t_span = (c_prepared.iloc[0, 0], c_prepared.iloc[-1, 0])
# T = int(c_prepared.iloc[-1, 0])
# t_step = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0]
# t_end = 1
# time = c_prepared.iloc[:, 0].values
# state_0 = c_prepared.iloc[0, 1:].values
# k = np.ones(X.shape[1])**10**(-12)
# N_k = len(k)

# kinetics = np.ones(X.shape[1])**10**(-12)


# Using Finite Difference


# USING solve_ivp

def reaction_network(t, state, *kinetics):
    # print('kinetics = ', kinetics)
    # print('state = ', state)
    mass_action_vector = np.array(kinetics)
    for r in range(N_k):
        # print('r =', r)
        # print(A[r,:])
        # print(np.prod( np.power(state, A[r,:]) ))
        # print('state =', state)
        mass_action_vector[r] *= np.prod( np.power( state, A[r,:]) )
    state_dot = X.dot(mass_action_vector)
    return state_dot

# write all reactions based on that of 1.


# states_predicted = solve_ivp(reaction_network, t_span, state_0, args=k, t_eval=time, method='RK45')
# save_at
# print(states_predicted)


# USING ode 

# r = ode(reaction_network)
# r.set_integrator('dopri5')
# r.set_initial_value(state_0, 0)
# r.set_f_params(*k)
# t1 = 1 # len(c_prepared.iloc[:, 0])
# dt = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0] 


# Y = [r.y]

# while r.successful() and r.t < t1:
#     r.integrate(r.t+dt)
#     Y.append(r.y)

# Y = np.array(Y)

# print(Y)
# print(states_predicted[:,1])





# Evolve network using finite difference method
def evolve_network_for_test( k_exp , x0, A, Q, t_step, t_end ):
    
    ''' Returns: predicted states'''
    # Define number of reactions 
    N_k = A.shape[0]

    # Copy initial conditions as current state
    state_current = x0.copy()

    # Set initial conditions as first entry of output
    states_predicted = np.array( state_current ) 

    # Loop through time
    for t in range( t_end ): 

        # print('t =', t)
        # print('\t state_current (before) = \n', state_current)
        # print('\t k_exp (before) = \n', k_exp)
        # k = k_exp
        mass_action_vector = np.power( 2, k_exp.copy() ) 

        # print('\t state_current (before) = \n', state_current[0:5])
        
        # Construct the mass action vector
        # mass_action_vector = k
        # print('MAV (before)', mass_action_vector[0:5] )
        
        for r in range(N_k):
            # print('r =', r)
            # print(A[r,:])
            # print(np.prod( np.power(state_current, A[r,:]) ))
            mass_action_vector[r] *= np.prod( np.power(state_current, A[r,:]) ) # Copy state_current to make it the same shape as A[:,:]

        # print('\t MAV (after) = ', mass_action_vector[0:5])

        # Compute the concetration vector for the next step, c_next = c(t + 1)  
        c_next = state_current + t_step * Q.dot(mass_action_vector)
       
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