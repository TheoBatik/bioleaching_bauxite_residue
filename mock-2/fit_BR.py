import time
# KINETIC FITTER
from kinetic_fitter import KineticFitter
kinf = KineticFitter()
import numpy as np
from scipy.optimize import basinhopping


# INPUT
kinf.set_stoichiometry( '2_A', '2_B' )
kinf.input_raw_data( '2_raw_data' )
kinf.check_consistency_across_inputs()
kinf.set_hidden_states()

# CLEAN DATA 
kinf.clean_measured_data()

# MODEL
kinf.set_initial_conditions()
kinf.set_evalutation_times()

# OPTIMISATION
kinf.evolve_net_method = 'Radau' # 'BDF' # 'LSODA' , 
n_epochs = 1
iters_per_epoch = 1
k_expo_0 = kinf.fetch_random_k_expo()
kinf.prepare_data_for_optimisation()
# t_start = time.time()
# kinf.fit_kinetics_by_basinhopping( n_epochs, iters_per_epoch, k_expo_0, display=True )
# t_end = time.time() 
# print('Time elapsed: ', t_end - t_start)


# Optimise kinetic parameters
def fit_kinetics_by_basinhopping( object, n_epochs, n_iters, k_expo, display=False):
        
    setattr(object, 'display', display)

    # Setup
    object.k_expo_tracker.append( k_expo )
    object.objective_tracker.append( object.objective( k_expo ) )
    
    N = 0
    T = 0.5 # temperature
    while N < n_epochs:
        
        if display:
            print('Epoch ', N)
        
        # Minimise globally
        a = basinhopping(object.objective, object.fetch_random_k_expo(), minimizer_kwargs=object.minimizer_kwargs, T=T,
                    niter=n_iters, disp=display, take_step=object.custom_hop, callback=None)#object.track_convergence)

        # Update 'temperature' and kinetics
        # k_expo = 
        # delta_obj = object.objective_tracker[-2] - object.objective_tracker[-1]
        # if delta_obj < 10e-4:
        #     T = 1/delta_obj
        #     k_expo = object.fetch_random_k_expo()
        # else:
        #     T = delta_obj
        #     k_expo = object.k_expo_tracker[-1] # global_minimizer.x

        # Update counter
        N += 1

    # Update attributes
    print( 'Results', a.x, a.fun )
    k_tracker = np.power( 2, object.k_expo_tracker )
    k_optimal = k_tracker[-1]
    lowest_objective = object.objective_tracker[-1]
    best_prediction = object.evolve_network( object.k_expo_tracker[-1] )
    iterations = np.arange( 0, len( k_tracker ) )
    setattr( object, 'k_tracker', k_tracker)
    setattr( object, 'k_optimal', k_optimal)
    setattr( object, 'lowest_objective', lowest_objective)
    setattr( object, 'best_prediction', best_prediction)
    setattr( object, 'iterations', iterations)
    setattr(object, 'display', False)

fit_kinetics_by_basinhopping(kinf, 1, 1, k_expo_0)

# RESULTS
kinf.results_to_csv()
kinf.plot_objective_by_iteration('Objective function value by iteration')
kinf.plot_kinetics_by_iteration('Kinetic parameter values by iteration number')
kinf.plot_predicted_vs_measured('Concetration of citric acid and Sc (aq) over time')

print(kinf.best_prediction)

attrs = vars(kinf)
for attr in attrs.items():
    print( attr )
