
# Setup 
from src.acid_production import AcidAniger
from src.utils import load_csv
import numpy as np
a = AcidAniger()

#------------------------------------------------------------------------------------------

# Load input
var_names = ['Biomass', 'Substrate', 'Citric acid', 'Oxalic acid', 'Gluconic acid']
a.input( 
    load_csv( csv_name='acid_production/measured_data' ),
    var_names
)

print('\nMeasured vars\n', a.vars_m)
print('Measured states\n', a.states_m)
print('Evaluation times\n', a.eval_times)

#------------------------------------------------------------------------------------------

# Set initial conditions, model parameters, and var_names

x0 = 4 # mg/L 
s0 = 125000#625000 # [substrate] at t=0 (sucrose) mg/L
p0 = [ a.states_m[0, i] for i in range(0,3) ] # mg/L
f0 = [x0, s0, *p0]
a.set_initial_conditions( f0 )

params_0 = ( 0.5, 0.01704, 1, 0.2, 0.2, 0.4 )



print( 'Initial conditions\n', a.f0 )

#------------------------------------------------------------------------------------------

# # Test solution with initial model params
# t = sorted( np.concatenate((
#         a.eval_times, 
#         np.logspace(-5, np.log2(a.eval_times[-1]), base=2) 
#     ))
# )
f = a.solve_system( params_0 ) #, t_eval=t )
print(f)

# cost = a.cost( params_0 )
# # print( 'Cost', cost )

# # f_model = a.cost( params_0 )
# # print( 'f_model', f_model )

#------------------------------------------------------------------------------------------

# Optimise parameters
optimal_params = a.optimise_basinhop( 
    params_0, 
    n_hops=1,
    display=True 
)
print( 'Optimal parameters\n', optimal_params )
f_optimal = a.solve_system( optimal_params )
print( 'Optimal output\n', f_optimal )

#------------------------------------------------------------------------------------------

# Save results
a.save_results( ignore=[] )#  s for s in a.var_names if s != 'Substrate' ] )

# # Fit solution to measured data
# # popt, pcov = a.fit( params_0 )

# import matplotlib.pyplot as plt
# print(t.shape)

# plt.figure()
# for i in range(0, 5):
#     plt.plot( t, f_optimal[i, :] )
# for i in range(0,3):
#     plt.plot( a.eval_times, a.states_m[:, i])
# plt.show()