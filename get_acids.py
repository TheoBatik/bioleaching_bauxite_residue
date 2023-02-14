
# Setup 
from src.acid_production import AcidAniger
from src.utils import load_csv
import numpy as np
a = AcidAniger()


# Load input
a.input( load_csv( csv_name='acid_production/measured_data' ) )
print('\nMeasured species\n', a.species_m)
print('Measured states\n', a.states_m)
print('Evaluation times\n', a.eval_times)


# Set initial conditions
x0 = 4 # mg/L
s0 = 625000 # [substrate] at t=0 (sucrose) mg/L
p0 = [ a.states_m[0, i] for i in range(0,3) ] # mg/L
f0 = [x0, s0, *p0]
a.set_initial_conditions( f0 )
print( 'Initial conditions\n', f0 )


# Test solution with random model params
t = np.linspace(0,50,100)
params_0 = [ 2.58, 0.1704, 1, 0.2, 3, 0.4 ] 
f_full = a.solve_system( t, params_0 )
f_reduced = a.solve_system_visible( t, params_0 )
print( f_reduced )


# Fit solution to measured data
popt, pcov = a.fit( params_0 )




import matplotlib.pyplot as plt

plt.figure()
for i in range(0, 3):
    plt.plot( t, f_reduced[:, i] )
plt.show()