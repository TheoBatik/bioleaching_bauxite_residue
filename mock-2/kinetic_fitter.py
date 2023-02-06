import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from datetime import datetime

class CustomBasinHop:
    def __init__(self, stepsize=1):
        self.stepsize = stepsize
    def __call__(self, k):
        s = self.stepsize
        # high = np.log2( 2 - 2**(-s) ) # Adjust upper step to account for k being exponentiated
        random_step = np.random.uniform(low=-s, high=s, size=k.shape)
        k += random_step
        return k


class KineticFitter:

    def __init__( self ):

        self.base_dir = os.getcwd()
        self.objective_tracker = []
        self.k_expo_tracker = []
        self.custom_hop = CustomBasinHop()
        self.evolve_net_method = 'LSODA'
        self.minimizer_kwargs = {"method": "L-BFGS-B"}
        self.k_expo_low = -1
        self.k_expo_high = 6
        self.display = True


    # INPUT
    
    # Function to load the .csv for the stoich. matrices + normalize each row    
    def load_csv( self, file_name ):
            matrix = []
            with open(f'{file_name}.csv', 'r') as file:
                for i, line in enumerate( file ):
                    if i == 0:
                        header = line.strip().split(',')
                        continue
                    row = [ int( item ) for item in line.strip().split(',') ]
                    row_max = max( row )
                    row_normalized = np.array( row ) / row_max
                    matrix.append( row_normalized )
            matrix = np.asarray( matrix )
            return matrix, header

    # Sets the stoichiometry
    def set_stoichiometry( self, csv_A, csv_B ):
        
        # Move to 'raw' data folder
        os.chdir( os.path.join(self.base_dir, 'data', 'raw') )

        # Define the stoich. matrices, A & B
        A, header_A = self.load_csv(csv_A)
        B, header_B = self.load_csv(csv_B)
       

        # Define the stoich. matrix, X = B - A
        X = B - A
        X = np.vstack( (X, -X) ) # stacks the coefficients of the reverse reactions (-X) onto those of the forward reactions (X)
        X = X.transpose() # such that N(columns) = N(reactions) & N(rows) = N(species)
        A = np.vstack( (A, B) )
        B = np.vstack( (B, A) )

        # Update attributes
        setattr(self, 'A', A)
        setattr(self, 'B', B)
        setattr(self, 'X', X)
        setattr(self, 'header_A', header_A)
        setattr(self, 'header_B', header_B)

        # Move back to base directory
        os.chdir( self.base_dir )

    # Inputs the raw data 
    def input_raw_data( self, csv_raw_data ):
        os.chdir( os.path.join(self.base_dir, 'data', 'raw') ) # Move to 'raw' data folder
        raw_data = pd.read_csv( csv_raw_data + '.csv' )
        header_raw = raw_data.columns[1:] # excl. first column = time
        setattr( self, 'raw_data', raw_data )
        setattr( self, 'header_raw', header_raw )
        os.chdir( self.base_dir ) # Move back to base directory

    # Checks that the species listed in all input headers match each other
    def check_consistency_across_inputs( self ):
        N_s = len( self.header_A ) # number of species
        for i in range( N_s ):
            consistent = bool( self.header_raw[i] == self.header_A[i] and self.header_raw[i] == self.header_B[i] )
            if not consistent:
                raise( f'Species mismatch on column {i}: ', self.header_raw[i], self.header_A[i], self.header_B[i] )
        # Set system dimensions
        N_k = self.X.shape[1] # number of reaction / kinetic parameters
        N_t = len( self.raw_data ) # number of time points
        setattr( self, 'N_s', N_s )
        setattr( self, 'N_k', N_k )
        setattr( self, 'N_t', N_t )

    # Store header indices for hidden states (prefixed 'H_')
    def set_hidden_states( self ):
        hidden_states = []
        for i, species in enumerate( self.header_A ):
            prefix = species[0:2]
            if prefix == 'H_':
                hidden_states.append(i)
        setattr( self, 'hidden_states', hidden_states )

    # Clean measured data
    def clean_measured_data( self ):
        
        # Copt raw data to be cleaned
        clean_data = self.raw_data

        # Convert micro-grams into miligrams, for each species listed
        columns = ['Sc_aqueous']
        for ree in columns:
            for i in range( self.N_t ):
                clean_data.loc[i, [ree]] *= 0.001 
        
        # Update attributes
        setattr( self, 'clean_data', clean_data )


    # MODEL 

    # Reaction network
    def reaction_network( self, t, state, *kinetics_exp ):
        mass_action = np.multiply( np.power( 2, kinetics_exp ), np.prod( np.power( np.clip(state, 0, None) , self.A[:,:] ), axis=1 ))
        state_dot = self.X.dot( mass_action )
        return state_dot
    
    # Set the initial conditions
    def set_initial_conditions( self ):
        clean_data_np = self.clean_data.to_numpy()
        state_0 = clean_data_np[0, 1:]
        setattr( self, 'state_0', state_0)

    # Sets the points in time at which the states are to be predicted + the time span
    def set_evalutation_times( self ):
        times = self.clean_data.loc[:, ['Day']].values[:, 0]
        t_span = ( times[0], times[-1] )
        setattr( self, 'times', times )
        setattr( self, 't_span', t_span )

    # Evolve netork / predict states
    def evolve_network( self,  k_exp ):
        result = solve_ivp( self.reaction_network, self.t_span, self.state_0, args=k_exp, t_eval=self.times, method=self.evolve_net_method)
        return result.y # .T


    # OPTIMISATION

    # Prepare data for the objective function
    def prepare_data_for_optimisation( self ):
        
        # Drop time column from clean data
        data_drop_days = self.clean_data.drop(['Day'], axis=1)
        
        # Delete hidden states
        visible_states = np.delete( data_drop_days.to_numpy(), self.hidden_states, 1 )
        
        # Normalise using the max value of each column
        visible_states_norm = visible_states.copy()
        n_columns = visible_states_norm.shape[1]
        column_maxes = []
        for i in range( n_columns ):
            column = visible_states_norm[:, i]
            column_max = column.max()
            column_norm = np.divide( column, column_max )
            visible_states_norm[:, i] = column_norm
            column_maxes.append( column_max )
        column_maxes = np.array( column_maxes )
        
        # Update attributes
        setattr( self, 'column_maxes', column_maxes )
        setattr( self, 'measured_visible_states_norm', visible_states_norm )

    # Objective function
    def objective( self, k_expo ):
               
        # Delete hidden states from the full set predicted
        predicted_visible_states = np.delete( self.evolve_network( k_expo ), self.hidden_states, 0 )

        # Normalise the visible states predicted (using the measured states' maxes)
        predicted_visible_states_norm = predicted_visible_states.T / self.column_maxes
        
        # Compute the sum over time of the squared discrepency between the predicted and measured states
        discrepency = predicted_visible_states_norm[:, :] - self.measured_visible_states_norm[:, :]
        error_squared = np.sum( np.multiply( discrepency, discrepency ) )

        # Add a penalty for crossing the bounds
        # k_too_low = bool(np.all(k_exp <= -2))
        # if k_too_low:
        #     penalty = 10**12
        #     error_squared += penalty

        return error_squared

    def track_convergence( self, k, f, accepted ):
        if abs(f - self.objective_tracker[-1]) < 1e-3: # and accepted 
            self.objective_tracker.append(f)
            self.k_expo_tracker.append(k)

    def fetch_random_k_expo( self ): 
        return np.around( np.random.uniform( low=self.k_expo_low, high=self.k_expo_high, size=self.N_k ), 2 )

    # Optimise kinetic parameters
    def fit_kinetics_by_basinhopping( self, n_epochs, n_iters, k_expo, display=False):
        
        setattr(self, 'display', display)

        # Setup
        self.k_expo_tracker.append( k_expo )
        self.objective_tracker.append( self.objective( k_expo ) )
        
        N = 0
        T = 0.5 # temperature
        while N < n_epochs:
            
            if display:
                print('Epoch ', N)
            
            # Minimise globally
            a = basinhopping(self.objective, self.fetch_random_k_expo(), minimizer_kwargs=self.minimizer_kwargs, T=T,
                        niter=n_iters, disp=display, take_step=self.custom_hop, callback=None)#self.track_convergence)

            # Update 'temperature' and kinetics
            # k_expo = 
            # delta_obj = self.objective_tracker[-2] - self.objective_tracker[-1]
            # if delta_obj < 10e-4:
            #     T = 1/delta_obj
            #     k_expo = self.fetch_random_k_expo()
            # else:
            #     T = delta_obj
            #     k_expo = self.k_expo_tracker[-1] # global_minimizer.x

            # Update counter
            N += 1

        # Update attributes
        print( 'Results', a.x, a.fun )
        k_tracker = np.power( 2, self.k_expo_tracker )
        k_optimal = k_tracker[-1]
        lowest_objective = self.objective_tracker[-1]
        best_prediction = self.evolve_network( self.k_expo_tracker[-1] )
        iterations = np.arange( 0, len( k_tracker ) )
        setattr( self, 'k_tracker', k_tracker)
        setattr( self, 'k_optimal', k_optimal)
        setattr( self, 'lowest_objective', lowest_objective)
        setattr( self, 'best_prediction', best_prediction)
        setattr( self, 'iterations', iterations)
        setattr(self, 'display', False)


    # RESULTS

    # Save results to .csv
    def results_to_csv(self, out_folder='results'):
        out_dir = os.path.join( self.base_dir, out_folder )
        objectives = self.objective_tracker[1:]
        kinetics = self.k_tracker
        k_optimal = kinetics[-1]
        np.savetxt( os.path.join(out_dir, 'objectives.csv'), objectives, delimiter=",")
        np.savetxt( os.path.join(out_dir, 'kinetics.csv'), kinetics, delimiter=",")
        np.savetxt( os.path.join(out_dir, 'k_optimal.csv'), k_optimal, delimiter=",")

    # Plot objective function values over iterations
    def plot_objective_by_iteration( self, title ):
        title += ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        iterations = self.iterations[1:] # remove initial estimate
        objective_tracker = self.objective_tracker[1:]
        fig = plt.figure()
        plt.plot( iterations, objective_tracker, 'o', linestyle='--', linewidth=1, markersize=0.1)
        plt.savefig( os.path.join( 'results', title ) )

    # Plot kinetic parameters by iteration
    def plot_kinetics_by_iteration( self, title ):
        title += ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig, axs = plt.subplots( 3, 2 )
        for i in range(3):
            axs[ i, 0 ].plot( self.iterations, self.k_tracker[ : , i], 'o', linestyle='--', linewidth=1, markersize=0.1 )
            axs[ i, 0 ].set_title( f'Kinetic parameter {i}' )
            axs[ i, 1 ].plot( self.iterations, self.k_tracker[ : , i + 3], 'o', linestyle='--', linewidth=1, markersize=0.1 )
            axs[ i, 1 ].set_title( f'Kinetic parameter {i + 3}' )
        fig.tight_layout()
        plt.savefig( os.path.join( 'results', title ) )

    # Plot predicted visible states against those measured
    def plot_predicted_vs_measured( self, title ):
        title += ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        fig, axs = plt.subplots( 2 )
        axs[0].plot( self.times, self.best_prediction[0, :], color='green', linestyle='--', linewidth=1, markersize=0 )
        axs[0].plot( self.times, self.clean_data[:, 0], color='blue', linestyle='-', linewidth=1, markersize=0 )
        axs[0].set_title( self.header_raw[0] )
        axs[1].plot( self.times, self.best_prediction[-2, :], color='green', linestyle='--', linewidth=1, markersize=0 )
        axs[1].plot( self.times, self.clean_data[:, -2], color='blue', linestyle='-', linewidth=1, markersize=0 )
        axs[1].set_title( str( self.header_raw[-2] ) )
        fig.tight_layout()
        plt.savefig( os.path.join( 'results', title ) )