# ORGANIC ACID PRODUCTION OF ASPERGILLUS NIGER


# Imports
from scipy.integrate import odeint, solve_ivp
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


    def ode_system( self, t, f, args ):
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

    # Using odeint
    def solve_full_system( self, t, *args ):
        '''
        Solves the ODE system for all variables 
        given the model parameters
        '''
        # if isinstance( t, float ) or isinstance( t, int ):
        #     t = [t]
        
        f_full = odeint(
            self.ode_system,
            self.f0,
            t,
            args=args,
            rtol=1e-13,
            atol=1e-12
        )

        return f_full

    # Using solve_ivp
    def solve_system( self, *params, t_eval=None ):
        '''
        Solves the ODE system for all variables 
        given the model parameters
        '''
        # if isinstance( t, float ) or isinstance( t, int ):
        #     t = [t]
        
        if t_eval is None:
            t_eval = self.eval_times

        print( 'Eval times' , t_eval )
        print( 't span', [t_eval[1], t_eval[-1]]  )

        sol = solve_ivp(
            self.ode_system, # system to solve
            [t_eval[0], t_eval[-1]], # time span
            self.f0, # initial conditions
            method='RK45',
            t_eval=t_eval,
            args=params, # parameters
            dense_output=False
        )

        return sol
    
    
    def set_times( self ):
        
        t_filler = np.logspace( 
            -3 , 
            np.log2( int(self.eval_times[-1] )), 
            num=150, 
            base=2
        )

        t_all = sorted( 
                np.concatenate((
                    t_filler, 
                    self.eval_times
                )) 
            )

        indices = []
        for i, t in enumerate(t_all):
            if t in self.eval_times:
                j = self.eval_times.index( t )
                indices.append( j )

        return indices
        

    def cost( self, params ):
        
        t_filler = np.logspace( 
            -3 , 
            np.log2( int(self.eval_times[-1] )), 
            num=150, 
            base=2
        )

        t = sorted( 
                np.concatenate((
                    t_filler, 
                    self.eval_times
                )) 
            )

        f_full = self.solve_full_system( t, params )
        f_reduced = np.delete( f_full, [0, 1], 1 )



        return f_reduced





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