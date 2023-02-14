# ORGANIC ACID PRODUCTION OF ASPERGILLUS NIGER


# Imports
from scipy.integrate import odeint
from scipy.optimize import curve_fit
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


    def ode_system( self, f, t, args ):
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


    def solve_system( self, t, *args ):
        '''
        Solves the ODE system for all variables 
        given the model parameters
        '''
        if isinstance( t, float ) or isinstance( t, int ):
            t = [t]
        
        f_full = odeint(
            self.ode_system,
            self.f0,
            t,
            args=args
        )
        return f_full

    def solve_system_visible( self, t, *args ):

        f_full = self.solve_system( t, *args )
        f_reduced = np.delete( f_full, [0, 1], 1 )
        
        return f_reduced



        # b0 = [X0, S0, DO0, P0]
        # t = np.linspace(0.01,55,55)
        # t0 = [0,0]
        # g = odeint(MICROBIAL,b0,t)


    def fit( self, params_0 ):
        
        f = self.solve_system_visible
        xdata = self.eval_times
        ydata = self.states_m
        popt, pcov = curve_fit(f, xdata, ydata, params_0)

        return popt, pcov