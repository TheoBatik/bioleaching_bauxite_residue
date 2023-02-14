#------------------------------------------------------------------------------------------
# ESTIMATING THE KINETIC PARAMETERS FOR SULFIDE MINERAL BIOLEACHING SYSTEM
#------------------------------------------------------------------------------------------

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
    load_csv( csv_name='measured_data_oxalic' ),
    fetch_initial_states( ropter.species, csv_name='c0_base_test_oxalic' ) 
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
    n_epochs=1,
    n_hops=2,
    atol=0.01, 
    rtol=0.001
    )
print( optimal_k )

#------------------------------------------------------------------------------------------

ropter.save_results(
    # eval_times = 
    #     # sorted( np.concatenate((
    # #     ropter.eval_times, 
    #     np.logspace(-5, -3, base=2),
    #     # ))
    # # ),
    #ignore=[],
    predicted=True, 
    measured=True
)

#------------------------------------------------------------------------------------------

# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)