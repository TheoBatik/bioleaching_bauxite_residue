
# Setup 
from src.acid_production import AcidAniger
from src.utils import load_csv
import numpy as np
a = AcidAniger()

#------------------------------------------------------------------------------------------

# Control 
optimise = True

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

# Set initial conditions, model parameters, and variable names
x0 = 10 # mg/L 
s0 = 625000 # [substrate] at t=0 (sucrose) mg/L # 125000#
p0 = [ a.states_m[0, i] for i in range(0,3) ] # mg/L
f0 = [x0, s0, *p0]
a.set_initial_conditions( f0 )
print( 'Initial conditions\n', a.f0 )



#------------------------------------------------------------------------------------------

# Optimise parameters
if optimise:
    params_0 = ( 
        8.664, # mumax, /day
        0.021447, # Ks, mg/L
        2.17680, # q, substrate consumption factor on dxdt
        0.29370, # r, substrate consumption factor on x
        0.5,
        0.01704, 
        1, 
        0.2, 
        0.2, 
        0.4
    )
    optimal_params = a.optimise_basinhop( 
        params_0, 
        n_hops=1,
        display=True 
    )
else:
    optimal_params = [ 
        5.77006406e+00, 1.24934596e-03, 3.23092737e+00, 7.84116789e-01,
        -3.61619392e-02, 5.88515867e-01, -5.88995094e-02, 3.74772579e-01,
        -1.82400320e-01, 1.05600231e+00
    ]
    a.popt = optimal_params

print( 'Optimal parameters\n', optimal_params )
print( 'Cost\n', a.cost( optimal_params ) )



# #------------------------------------------------------------------------------------------

# Save results

# Set model evaluation times
t = np.linspace( 0, int(a.eval_times[-1]), 100 )

# All quantities
a.save_results( 
    ignore=[], 
    eval_times=t,
    measured=True,
    predicted=True,
    plot_name='(all quantities)'
)

# Acid only
a.save_results( 
    ignore=[ 'Substrate', 'Biomass' ],
    eval_times=t,
    measured=True,
    predicted=True,
    plot_name='(acids only)'
)

# Substrate only
a.save_results( 
    ignore=[s for s in a.var_names if s != 'Substrate' ],
    eval_times=t,
    measured=False,
    predicted=True,
    plot_name='(substrate)'
)

# Biomass only
a.save_results( 
    ignore=[s for s in a.var_names if s != 'Biomass' ],
    eval_times=t,
    measured=False,
    predicted=True,
    plot_name='(biomass)'
)
