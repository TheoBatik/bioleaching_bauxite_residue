#------------------------------------------------------------------------------------------
# ESTIMATING THE KINETIC PARAMETERS FOR SULFIDE MINERAL BIOLEACHING SYSTEM
#------------------------------------------------------------------------------------------

# Setup
from src.rate_optimiser import RateOptimiser
import numpy as np
from src.utils import load_csv, fetch_initial_states
ropter = RateOptimiser( reversible=False )
ropter.set_system()

#------------------------------------------------------------------------------------------

# Control

# Data
measured_data = 'leaching/measured_data_test'
ic_data = 'leaching/c0_base_test'

# Optimisation
n_epochs = 1
n_hops = 1
atol = 1e-3
rtol = 1e-4

#------------------------------------------------------------------------------------------

print(ropter.species)

# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)

#------------------------------------------------------------------------------------------

ropter.input( 
    load_csv( csv_name=measured_data ),
    fetch_initial_states( ropter.species, csv_name=ic_data ) 
) 

#------------------------------------------------------------------------------------------

# Checks 
print('Measured species', ropter.species_m)
print('Measured states', ropter.states_m)
print('All species', ropter.species)
print('Indices of hidden states', ropter.ihs)
print('Normalised measured states', ropter.states_nm)


#------------------------------------------------------------------------------------------

# Optimisation
optimal_k = ropter.optimise( 
    n_epochs=n_epochs,
    n_hops=n_hops,
    atol=atol, 
    rtol=rtol
    )
print( optimal_k )

#------------------------------------------------------------------------------------------

# Save results

# Evaluation times
# t = sorted( np.concatenate((
#         ropter.eval_times, 
#         np.logspace(-2, -0.1, base=2, num=6),
#         ))
#     )
t = np.linspace( 0, int(ropter.eval_times[-1]), 30 )


# ...
ropter.save_results(
    eval_times=t,
        # 
    #ignore=[],
    predicted=True, 
    measured=True
)

#------------------------------------------------------------------------------------------

# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)