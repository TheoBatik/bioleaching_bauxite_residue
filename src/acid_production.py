# ORGANIC ACID PRODUCTION OF ASPERGILLUS NIGER


# Imports
from src.rate_optimiser import custom_hop
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
import numpy as np


# Fixed model parameters
umax = 0.361 # /h
Ks = 21.447 # g/L
q = 2.17680
r = 0.29370

# Initial guess of fixable parameters
class AcidAniger:
    
    def input( self, measurements, var_names ):

        # Set input attributes
        setattr( self, 'states_m', measurements[0][:, 1:] ) # measured states
        setattr( self, 'vars_m', measurements[1][1:] ) # measured variables
        setattr( self, 'eval_times', measurements[0][:, 0] ) # times at which to evaluate the soln
        setattr( self, 'var_names', var_names ) # times at which to evaluate the soln

        # Set maxium of the measured states
        setattr( self, 'max_measured', np.max( self.states_m ) )

        # Normlise the measured states
        setattr( self, 'states_nm', self.states_m / self.max_measured )


    def ode_system( self, t, f, *args ):
        ''' 
        System of differential equations for:
        1) Biomass production, x (Monod dynamics assumed)
        2) Substrate consumption, s
        3) Organic acid production, p
            pci -> citric acid
            pox -> oxalic acid
            pgl -> gluconic acid
        '''
        # Element-wise unpacking of vectorised solution, f 
        x = f[0]
        s = f[1]

        if s <= 0:
            return np.zeros(5)
        else:
            # Biomass production rate
            dxdt = args[0]*(s/(args[1]*x+s)) * x

            # Substrate consumption rate
            dsdt = - args[2] * dxdt - args[3] * x

            # Acid production rates
            dpdt = [ args[i] * dxdt + abs(args[i+1]) * x for i in [4, 6, 8] ]
            
            # Return ODE system
            return [dxdt, dsdt, *dpdt]


    def set_initial_conditions( self, f0 ):
        setattr( self, 'f0', f0 )


    # Using solve_ivp
    def solve_system( self, params, t_eval=None ):
        '''
        Solves the ODE system for all variables 
        given the model parameters
        '''
        
        if t_eval is None:
            t_eval = self.eval_times

        f = solve_ivp(
            self.ode_system, # system to solve
            [t_eval[0], t_eval[-1]], # time span
            self.f0, # initial conditions
            method='Radau', # Solver method
            t_eval=t_eval, # Soln evaluation times
            args=params, # parameters
            dense_output=False
        ).y

        return f
        

    def cost( self, params ):
        
        states_p = np.transpose( self.solve_system( params ) ) # predicted states
        states_nvp = np.delete( states_p, [0, 1], 1 ) / self.max_measured # normalised visible predicted states 
        del states_p

        # Calculate the net error: sum (over time) of the discrepencies squared
        discrepency = states_nvp[1:, :] - self.states_nm[1:, :]
        net_error = np.sum( np.multiply( discrepency, discrepency ) )

        return net_error 

    
    def optimise_basinhop( self, params, n_hops=3, display=False ):

        optimal_params = basinhopping(
            self.cost, 
            params, 
            minimizer_kwargs={'method':'L-BFGS-B'},
            # T=None,
            niter=n_hops, 
            disp=display, 
            take_step=custom_hop,
            callback=None
        ).x 

        # Get optimal predicted states
        states_p = self.solve_system( optimal_params )
        
        # Set results as attributes
        setattr( self, 'popt', optimal_params )
        setattr( self, 'states_p', states_p )

        return optimal_params
        

    def save_results( 
        self, 
        eval_times=None, 
        ignore=None, 
        predicted=True, 
        measured=True,
        plot_name='',
        plot_name_stem='Organic acid production by A. niger:'
        # timestamp=False
        ): 
        
        # Update predicted states, if required
        if eval_times is None:
            # Use existing attributes
            eval_times = self.eval_times
            states_p = self.states_p
        else:
            # Given evaluation times, derive new prediction 
            states_p = self.solve_system( self.popt, t_eval=eval_times )
        
        # Plot predicted and measured states
        if ignore is None:
            ignore = [ s for s in self.var_names if s not in self.vars_m]
        colours = plt.cm.rainbow(np.linspace(0, 1, len(self.var_names)))
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax in axes:
            # Plot predicted states
            for i, s in enumerate( self.var_names ):
                if s not in ignore:
                    if predicted:
                        ax.plot( 
                            eval_times,
                            states_p[i, :],
                            linestyle='dashed',
                            label=s + ' (predicted)',
                            c=colours[i]
                    )
                    if measured and s in self.vars_m:
                        j = self.vars_m.index(s)
                        ax.plot( 
                            self.eval_times,
                            self.states_m[:, j],
                            linestyle = 'None',
                            marker='.',
                            ms=6,
                            label=s + ' (measured)',
                            c=colours[i]
                        )

        # Set legend and axes' lables

            _ = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            _ = ax.set_xlabel('Time (days)')
            _ = ax.set_ylabel('Concentration (mg/L)')
        # Adjust titles and scales
        _ = axes[1].set_xscale('log')
        _ = axes[1].set_yscale('log')
        axes[0].set_title('Normal scale', loc='left')
        axes[1].set_title('Log scale', loc='left')

        # Tidy and Save
        _ = axes[0].legend().remove()
        suptitle = plot_name_stem + ' predicted and measured concentrations over time ' + plot_name 
        fig.suptitle( suptitle, fontsize=16 )
        _ = fig.tight_layout()
        save_at = 'results/acid_production/plots/' + plot_name_stem + ' ' + plot_name + '.png'
        _ = fig.savefig( save_at, dpi=72 )
    #     

    #     # Save results to .csv
    #     print( 'Predicted states', states_p.yout )
    #     np.savetxt('results/states/predicted_states.csv', states_p.yout, delimiter=',')
    #     np.savetxt('results/states/vars.csv', self.vars, fmt='%s', delimiter=',')
    #     np.savetxt('results/states/eval_times.csv', states_p.xout, delimiter=',')
    #     np.savetxt('results/kinetics/optimal_rate_params.csv', self.optimal_rate_params, delimiter=',')