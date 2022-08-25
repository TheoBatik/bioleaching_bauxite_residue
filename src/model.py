from curses import init_color
from mimetypes import init
from prepare_input import X, A, c_prepared
import numpy as np

# Set inputs
c_0 = c_prepared.iloc[0, 1:].values
# print( 'c0 = ', c_0)
T = c_prepared.iloc[-1, 0]
delta_t = 0.00001 #T - c_prepared.iloc[-2, 0]
# print('A = ', A)
# Kinetic parameters, k,
# initial conditions, c_0
# Number of times at which predictions are made, N_time
# Stoichmetric matrix, X, A


def check_input_dimensions(number_species, number_reactions, stoich_X, stoich_A):
    if not (number_species == stoich_X.shape[0] and number_species == stoich_A.shape[1]):
        raise('The dimension of the initial conditions must match that of the stoichiometric matrices.')
    if not (number_reactions == stoich_X.shape[1] and number_reactions == stoich_A.shape[0]):
        raise('The dimension of the kinetics vector must match that of the stoichiometric matrices.')


def predict_c(init_conditions, kinetics, stoich_X=X, stoich_A=A, time_space=delta_t, time_span=T):
    
    ''' Return: predicted concentrations'''
    
    N_c = init_conditions.shape[0] # number of species
    N_k = len(kinetics) # number of reactions
    N_t = int(time_space * time_span)

    check_input_dimensions(N_c, N_k, stoich_X, stoich_A)

    # Set initial conditions as first entry of output
    c_predicted = np.array(init_conditions)

    # Derive total number of points in time
    

    # FORWARD STEP

    # Loop through time
    for t in range(4): # N_t

        print('t =', t)
        # print('\t init_conditions (before) = \n', init_conditions)
        # print('\t kinetics (before) = \n', kinetics)
        # k = kinetics
        mass_action_vector = kinetics.copy()

        print('\t init_conditions (before) = \n', init_conditions[0:5])
        
        # Construct the mass action vector
        # mass_action_vector = k
        # print('MAV (before)', mass_action_vector[0:5] )
        
        for r in range(N_k):
            # print('r =', r)
            # print(A[r,:])
            # print(np.prod( np.power(init_conditions, A[r,:]) ))
            mass_action_vector[r] = np.prod( np.power(init_conditions, A[r,:]) )

        print('\t MAV (after) = ', mass_action_vector[0:5])

        # Compute the concetration vector for the next step, c_next = c(t + 1)  
        c_next = init_conditions + delta_t * X.dot(mass_action_vector)

        # Impose lower bound of zero on c_next
        min_ic = 0
        max_ic = max(c_next)
        c_next = np.clip(c_next, min_ic, max_ic)

        # print('\t c_next = ', c_next)

        # # Stack c_next ontp c_predicted 
        c_predicted = np.vstack((c_predicted, c_next))

        # # Set c_next as the initial condition for the next step
        init_conditions = c_next
        
        print('\t init_conditions = \n', init_conditions[0:5])
        # print('\t kinetics (after) = \n', kinetics)

        print('\n')

    return c_predicted

        
k = np.ones(X.shape[1])**10**(-12)

c_predicted = predict_c(c_0, k)

print('\n c_predicted = \n', c_predicted)

