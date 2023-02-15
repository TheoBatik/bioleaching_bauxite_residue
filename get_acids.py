
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


# # Set initial conditions
x0 = 4 # mg/L
s0 = 125000#625000 # [substrate] at t=0 (sucrose) mg/L
p0 = [ a.states_m[0, i] for i in range(0,3) ] # mg/L
f0 = [x0, s0, *p0]
a.set_initial_conditions( f0 )
print( 'Initial conditions\n', a.f0 )


# # Test solution with random model params
params_0 = ( 0.5, 0.01704, 1, 0.2, 0.2, 0.4 )
f = a.solve_system( params_0 ).y
print(f.shape)

# f_model = a.cost( params_0 )
# print( 'f_model', f_model )


# Fit solution to measured data
# popt, pcov = a.fit( params_0 )




import matplotlib.pyplot as plt
t = a.eval_times
print(t.shape)


plt.figure()
for i in range(0, 5):
    plt.plot( t, f[i, :] )
# for i in range(0,3):
#     plt.plot( a.eval_times, a.states_m[:, i])
plt.show()