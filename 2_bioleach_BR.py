#------------------------------------------------------------------------------------------
# ESTIMATING THE KINETIC PARAMETERS FOR SULFIDE MINERAL BIOLEACHING SYSTEM
#------------------------------------------------------------------------------------------

# Setup
from src.rate_optimiser import RateOptimiser
import numpy as np
from src.utils import load_csv, fetch_initial_states
from src.stoichiometry import reactants, products
ropter = RateOptimiser( reversible=False )
ropter.set_system( reactants, products )

#------------------------------------------------------------------------------------------

# Control

# Data
measured_data = 'leaching/measured_data_test'
ic_data = 'leaching/c0_base_test'

# Optimisation
optimise = True
n_epochs = 1
n_hops = 1
atol = 1e-2
rtol = 1e-3

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
print('Measured species\n', ropter.species_m)
print('Measured states\n', ropter.states_m)
print('All species\n', ropter.species)
print('Indices of hidden states\n', ropter.ihs)
print('Normalised measured states\n', ropter.states_nm)


#------------------------------------------------------------------------------------------

# Optimisation
if optimise:
    optimal_k = ropter.optimise( 
        n_epochs=n_epochs,
        n_hops=n_hops,
        atol=atol, 
        rtol=rtol
    )
else:
    optimal_k = [
        -0.16545599,  -2.28359546, 1.42707124,  10.16995791, -2.33996827,
        -12.35152712,  -2.91383569,  -0.47316908,  11.02728277,  -0.8393299,  -6.40311828
    ]
    setattr( ropter, 'optimal_rate_params', optimal_k )
print( optimal_k )

#------------------------------------------------------------------------------------------

# Save results

# Evaluation times
# t = sorted( np.concatenate((
#         ropter.eval_times, 
#         np.logspace(-2, -0.1, base=2, num=6),
#         ))
#     )
t = np.linspace( 0, int(ropter.eval_times[-1]), 100 )


# All quantities
ropter.save_results(
    eval_times=t,
    predicted=True, 
    measured=True,
    plot_name='(all measured quantities)'
)

# REE's
ropter.save_results(
    eval_times=t,
    ignore=[ s for s in ropter.species if s not in ['Sc3+', 'Y3+'] ],
    predicted=True, 
    measured=True,
    plot_name='(REE\'s)'
)

# IC's
ics = [ 'Fe3+', 'Al3+', 'Ti4+', 'Ca2+']
ropter.save_results(
    eval_times=t,
    ignore=[ s for s in ropter.species if s not in ics ],
    predicted=True, 
    measured=True,
    plot_name='(IC\'s)'
)

# Individual plots
elements = [ 'Fe3+', 'Al3+', 'Ti4+', 'Ca2+', 'Sc3+', 'Y3+']
for element in elements:
    ropter.save_results(
        eval_times=t,
        ignore=[ s for s in ropter.species if s not in [ element ] ],
        predicted=True, 
        measured=True,
        plot_name='(' + element + ')'
    )

#------------------------------------------------------------------------------------------

# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)