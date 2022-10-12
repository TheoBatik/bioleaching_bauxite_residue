
# KINETIC FITTER
from kinetic_fitter import KineticFitter
kinf = KineticFitter()

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
iters_per_epoch = 25
k_expo_0 = kinf.fetch_random_k_expo()
kinf.prepare_data_for_optimisation()
kinf.fit_kinetics_by_basin_hopping( n_epochs, iters_per_epoch, k_expo_0, display=True )

# RESULTS
kinf.plot_objective_by_iteration()
kinf.plot_kinetics_by_iteration()
kinf.plot_predicted_vs_measured()

print(kinf.best_prediction)

attrs = vars(kinf)
for attr in attrs.items():
    print( attr )
