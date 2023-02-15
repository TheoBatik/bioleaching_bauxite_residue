# ORGANIC ACID PRODUCTION OF ASPERGILLUS NIGER


# Imports
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# from scipy.optimize import curve_fit
import numpy as np


# Fixed model parameters
umax = 0.361 # /h
Ks = 21.447 # g/L
q = 2.17680
r = 0.29370

# Initial guess of fixable parameters
class AcidAniger:
    
    def input( self, measurements ):

        # Set input attributes
        setattr( self, 'states_m', measurements[0][:, 1:] ) # measured states
        setattr( self, 'species_m', measurements[1][1:] ) # measured species
        setattr( self, 'eval_times', measurements[0][:, 0] ) # times at which to evaluate the soln

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

        # Biomass production rate
        u = umax*(s/(Ks*x+s))
        dxdt = u*x
        # Substrate consumption rate
        rs = -q * dxdt - r * x
        dsdt = rs
        # Acid production rates
        dpdt = [ args[i] * dxdt + args[i+1] * x for i in [0, 2, 4] ]
        
        # ODE system
        dfdt = [dxdt, dsdt, *dpdt]
        return dfdt


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
            method='RK45',
            t_eval=t_eval,
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


    def optimise( self, params_0, display=False ):

        optimal_params = minimize(self.cost, params_0, method='L-BFGS-B' ) #, tol=1e-6)
        
        if display:
            print( 'Cost function value at:', self.cost( optimal_params ) )

        return optimal_params



    # def fit( self, params_0 ):
        
    #     f = self.solve_flat_visible_system
            
    #     # xdata = np.tile( self.eval_times, 3 ) 
    #     ydata = self.states_m.flatten()
    #     xdata = []
    #     # for i in range(0,3):
            
    #     popt, pcov = curve_fit(
    #         f, 
    #         xdata=xdata,
    #         ydata=ydata,
    #         p0=params_0
    #     )

    #     return popt, pcov