#------------------------------------------------------------------------------------------
# ESTIMATING THE KINETIC PARAMETERS FOR SULFIDE MINERAL BIOLEACHING SYSTEM
#------------------------------------------------------------------------------------------

import numpy as np

# Setup

from src.rate_optimiser import RateOptimiser
from src.utils import load_csv, fetch_initial_states

ropter = RateOptimiser( reversible=False )
ropter.set_system()

#------------------------------------------------------------------------------------------

print(ropter.species)

# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)

#------------------------------------------------------------------------------------------

ropter.input( 
    load_csv( csv_name='measured_data' ),
    fetch_initial_states( ropter.species, csv_name='c0_base' ) 
) 
#------------------------------------------------------------------------------------------

# Checks 
print('species_m', ropter.species_m)
print('states_m', ropter.states_m)
print('species', ropter.species)
print('i hidden states', ropter.ihs)
print('states_nm', ropter.states_nm)


#------------------------------------------------------------------------------------------

# Optimisation

optimal_k = ropter.optimise( n_epochs=1, n_hops=1 )
print( optimal_k )
ropter.save_results( ) # eval_times=np.logspace(-15, -12) )


# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)