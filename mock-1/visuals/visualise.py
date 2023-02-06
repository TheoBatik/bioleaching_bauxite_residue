from src.input import X, A_header, days
from src.model import evolve_network
import numpy as np
import matplotlib.pyplot as plt



print(days)

# # Optimisation Inputs

# t_span = (c_prepared.iloc[0, 0], c_prepared.iloc[-1, 0])
# T = int(c_prepared.iloc[-1, 0])
# t_step = 0.01 # c_prepared.iloc[-1, 0] - c_prepared.iloc[-2, 0]
# t_end = 1
# time = c_prepared.iloc[:, 0].values
# state_0 = c_prepared.iloc[0, 1:].values
N_k = X.shape[ 1 ]
N_c = X.shape[ 0 ]
k_0 = np.ones( N_k ) * 10 ** (-4)
# k_bounds = [ ( 500, 1000) for _ in range(N_k) ]
# k_lower = np.zeros( N_k )
# k_upper = np.full( N_k, 1000 )
# k_converge = [ k_0 ]
# obj_function_mins = [ objective( k_0 ) ]


print('Species: ', A_header)

species = [0, 5, 7, 9]



states_predicted = evolve_network(k_0)


fig, axs = plt.subplots( len(species) )
fig.suptitle('Predicted concentrations over time')

for i, s in enumerate(species):
    axs[i].plot( days, states_predicted[ : , s ] )
    axs[i].set_title(A_header[s])

plt.savefig('states_predicted.png')



