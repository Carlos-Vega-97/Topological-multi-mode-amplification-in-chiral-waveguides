#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:40:17 2023

@author: carlos
"""

import numpy as np
import scipy.linalg as LA
import scipy.integrate as integrate

class GaussianSystem:
    """
    This class implements several methods for describing a driven-dissipative
    bosonic lattice. In particular, it describes a set of N bosonic modes
    coupled to a chiral multi-mode waveguide in the linear regime.
    
    Methods include a routine to find steady state correlations and topological
    features of the doubled-Hamiltonian.
    """
    
    def __init__(self, N):
        self.N = N  
        
    def create_params(self, Gammas, kl_s, pump, g_s, l_k, Delta=0, g_c=0):
        """
        Creates a dictionary 'params' with all the relevant parameters of the
        system: 
            
            - Gammas: List of decay rates to each chiral mode
            - kl_s: List of momenta of each chiral mode
            - pump: Amplitude of incoherent pump
            - g_s: Parametric driving
            - positions: List of positions of the bosonic modes
        """
        params = {}
        params['Gammas'] = Gammas
        params['kl_s'] = kl_s
        params['pump'] = pump
        params['g_s'] = g_s
        params['l_k'] = l_k
        params['Delta'] = Delta
        params['g_c'] = g_c
        
        return params
         
        
    def get_H_eff(self, params, positions=None):
        """
        Retrieves the matrix of the non-Hermitian Hamiltonian describing the
        collective interaction and dissipation induced in a system of N modes
        coupled to a chiral multi-mode waveguide.
        
        H_nh = Sum_ij (J_ij - iGamma_ij) b⁺i b_j
    
        Parameters
        ----------
        Gamma : list
            Loss rates
        ks : list
            Momenta of waveguide modes
        positions: list
            Emitter positions

        Returns
        -------
        H_eff : Matrix
            Effective Hamiltonian matrix

        """
        
        # Retrieve parameters from input dictionary
        Gammas = params['Gammas']
        kl_s = params['kl_s']
        l_k = params['l_k']
        Delta = params['Delta']
        
        # Get positions
        if positions==None:
            positions = np.arange(self.N)
        
        # Initialize matrix
        H_eff = np.zeros((self.N, self.N), dtype=complex)
        
        # Declare diagonal terms
        H_eff += np.diag(-(1j/2)*np.sum(Gammas)*np.ones((self.N)))
        
        # Add detuning
        H_eff += np.diag(Delta*np.ones((self.N)))
        
        # Turn lists into array to allow operations
        Gammas, kl_s = np.array(Gammas), np.array(kl_s)  
        
        # Declare non-diagonal terms
        for i in range(self.N):
            for j in range(i+1, self.N):
                H_eff[i,j] = -1j*np.sum(Gammas*np.exp((1j*kl_s-1/l_k)*np.abs((positions[i]-positions[j]))))
            
        return H_eff
    
    def get_H_nh(self, params):
        """
        Gets the non-Hermitian Hamiltonian determining the system evolution.
        It comprises the collective interactions and dissipation induced
        by the bath together with the effect of a incoherent local pump and
        loss

        Parameters
        ----------
        params : Dictionary
               System parameters

        Returns
        -------
        H_nh : Matrix
             Non-Hermitian matrix

        """
        # Retrieve pump and parametric driving from parameters dictionary
        pump= params['pump']
        g_s = params['g_s']
        
        # Define effective Hamiltonian
        H_eff = self.get_H_eff(params)
        
        # Define pump and parametric driving terms
        pump_term = 1j*(pump/2)*np.eye(self.N)
        K_term = g_s*np.eye(self.N)
        
        # Initialize non-Hermitian Hamiltonian matrix
        H_nh = np.zeros((2*self.N, 2*self.N), dtype=complex)
        
        # Diagonal terms (a-a & a^dagger-a^dagger couplings)
        H_nh[:self.N, :self.N] = pump_term + H_eff
        H_nh[self.N:, self.N:] = pump_term - H_eff.conj()
        
        # Off-diagonal terms (parametric driving i.e. a-a^dagger coupling)
        H_nh[:self.N, self.N:] = K_term
        H_nh[self.N:, :self.N] = -K_term.conj()
        
        return H_nh
    
    
    def get_H_nh_k_space(self, k, params):
        """
        Gets the 2x2 matrix representing the non-Hermitian Hamiltonian acting
        over the Nambu spinor (b_k, b^dagger_k) in momentum space, for our
        chiral-multimode scenario
        
        Parameters
        ----------
        k: float
           Momentum at which H_nk(k) is evaluated
        
        params : Dictionary
               System parameters

        Returns
        -------
        H_nh : Matrix
             2x2 Non-Hermitian matrix
        
        """
        
        # Retrieve parameters from input dictionary
        pump = params['pump']       # Incoherent pump amplitude
        Gammas = params['Gammas']   # Decay rates onto each mode
        kl_s = params['kl_s']       # Momentum of each mode
        g_s = params['g_s']         # Parametric driving
        l_k = params['l_k']         # Dissipative length
        Delta = params['Delta']     # Parametric drive detuning
        g_c = params['g_c']         # Collective parametric driving
        
        # Convert to numpy arrays to allow operations
        Gammas, kl_s = np.array(Gammas), np.array(kl_s)
    
        # Initialize matrix
        H_nh = np.zeros((2,2), dtype=complex)
        
        # Parametric driving
        H_nh[0,1] = g_s + 2*g_c*np.cos(k)
        H_nh[1,0] = -np.conjugate(g_s) -2*np.conjugate(g_c)*np.cos(k)
        
        # Incoherent pump
        H_nh[0,0]= 1j*pump/2
        H_nh[1,1]= 1j*pump/2
        
        # Local losses
        H_nh[0,0]+= -1j*np.sum(Gammas/2)
        H_nh[1,1]+= -1j*np.sum(Gammas/2)
        
        # Collective losses
        H_nh[0,0] += -1j*np.sum(Gammas*np.exp(1j*(k+kl_s)-1/l_k)/(1-np.exp(1j*(k+kl_s)-1/l_k)))
        H_nh[1,1] += -1j*np.sum(Gammas*np.exp(1j*(k-kl_s)-1/l_k)/(1-np.exp(1j*(k-kl_s)-1/l_k)))
        
        # Parametric drive detuning
        H_nh[0,0] += Delta
        H_nh[1,1] += -Delta
            
        return H_nh
    
        
    def get_dynamical_matrix(self, params):
        """
        Retrieves the dynamical matrix for the Nambu spinor
        X = (<b_1>,...,<b_N>,<b_1⁺>,...,<b_N⁺>)^T. It is equal to
        -iH_nh:
            
        (dX/dt) = AX = (-iH_nh)X

        Parameters
        ----------
        pump : Float 
            Amplitude of incoherent pump
        
        H_eff : Matrix
            Effective Hamiltonian
            

        Returns
        -------
        A : Dynamical matrix (dim. 2N x 2N)
        """
        
        A = -1j*self.get_H_nh(params)
        
        return A
    
    def get_steady_state_coherences(self, A, f):
        """
        For the steady state, 
        
              d/dt X_j = A_jl X_l -f = 0  ----> <b>_ss = A^-1 f

        Parameters
        ----------
        A : Dynamical matrix
        f : Array
            Coherent driving vector

        Returns
        -------
        b_ss : Array
               Coherences in the steady state

        """
        # Define coherent driving vector
        v = np.zeros((2*self.N))
        v[:self.N] = f
        v[self.N:] = f.conj()
        
        # Solve equation inverting dynamical matrix
        b_ss = np.linalg.inv(A) @ v
        
        return b_ss 
    
    def get_steady_state_covariance_matrix(self, A, pump, H_eff):
        """
        Gets the covariance matrix Theta in the steady state, obtained by
        solving the following Lyapunov equation
        
                   A Theta + Theta A^dagger + F = 0

        Parameters
        ----------
        A : Dynamical matrix
        F : Matrix
            Encodes the dissipator terms

        Returns
        -------
        Theta_ss : Matrix
                   Covariance matrix in the steady state

        """
        # Initialize matrix of dissipation-induced dynamics in the correlators
        F = np.zeros((2*self.N, 2*self.N))
        
        # Pump
        F[:self.N, :self.N] = pump*np.eye(self.N)
        
        # Loss
        F[self.N:, self.N:] = np.imag(H_eff)
        
        # Solve Lyapunov equation
        Theta_ss = LA.solve_continuous_lyapunov(A.conjugate(), -F)
        
        return Theta_ss
        
    def get_doubled_Hamiltonian(self, H_nh):
        """
        Builds the doubled Hamiltonian, whose topological non-triviality is
        related to steady-state amplification:
            
            H = (H_nh sigma_+) + (H_nh^dagger sigma_-) 
    
        Parameters
        ----------
        H_nh : Dynamical matrix

        Returns
        -------
        H : Doubled Hamiltonian
        """
        dim = H_nh.shape[0]
        H = np.zeros((2*dim, 2*dim), dtype=complex)
        H[:dim, dim:] = H_nh
        H[dim:, :dim] = H_nh.conj().transpose()
        
        return H
    
    def compute_winding_scalar(self, params, n=1000):
        """
        Computes the winding number associated to a complex function
        h(k). It is the number of times the vector (Re(h(k)), Im(h(k)))
        winds around the origin as k swipes the whole Brillouin Zone.
                      _                  
                 1   |        h'(k)
        nu = --------| dk   -------
              2pi i _|        h(k)
                       BZ
        
        Parameters
        ----------
        params : Dictionary
               System parameters
        
        n : integer
           Brillouin Zone discretization

        Returns
        -------
        nu: integer
            Winding number

        """
        # Set a step in momentum space
        delta_k = 2*np.pi/n
        
        # Define the integrand of the winding integral
        def get_f(k):
            
            # Spectrum
            h = self.get_H_nh_k_space(k, params)[0, 0]
            
            # Derivative
            partial_h = (self.get_H_nh_k_space(k+delta_k, params)[0, 0] 
                         - self.get_H_nh_k_space(k-delta_k, params)[0, 0])/(2*delta_k)
            
            return (1/h)*partial_h
        
        # Separate real and imaginary parts of the integrand
        real_f = lambda k: np.real(get_f(k))
        imag_f = lambda k: np.imag(get_f(k))
        
        # Perform integral
        nu = (1/(2*np.pi*1j))*integrate.quad(real_f, -np.pi, np.pi)[0]
        nu += (1/(2*np.pi))*integrate.quad(imag_f, -np.pi, np.pi)[0]
            
        return np.round(np.real(nu))
    
    def compute_winding_matrix(self, params, n=1000):
        """
        Computes the winding number from the doubled Hamiltonian
        matrix H as:
                      _                  
                 1   |                       dH
        W_1 = -------| dk   Tr[ tau_z  H⁻1 ------]
              4pi i _|                       dk
                       BZ
        
        """
         # Set a step in momentum space
        delta_k = 2*np.pi/n
        
        # Declare
        tau_z = np.array(np.diag([-1,-1,1,1]), dtype = complex)
        
        # Define integrand function
        def get_F(k):
            
            # Non-Hermitian Hamiltonian
            H_nh = self.get_H_nh_k_space(k, params)
           
            # Its derivative
            partial_H_nh = (self.get_H_nh_k_space(k+delta_k, params)
                         - self.get_H_nh_k_space(k-delta_k, params))/(2*delta_k)
            
            # Double degree of freedom
            H = self.get_doubled_Hamiltonian(H_nh)
            partial_H = self.get_doubled_Hamiltonian(partial_H_nh)
            
            return np.trace(tau_z @ np.linalg.inv(H) @ partial_H)
        
        # Separate real and imaginary parts of the integrand
        real_F = lambda k: np.real(get_F(k))
        imag_F = lambda k: np.imag(get_F(k))
        
        # Perform integral
        W1 = (1/(4*np.pi*1j))*integrate.quad(real_F, -np.pi, np.pi)[0]
        W1 += (1/(4*np.pi))*integrate.quad(imag_F, -np.pi, np.pi)[0]
        
        return np.round(np.real(W1))
    
    def is_unstable(self, params):
        """
        Is the system unstable?
        
        Returns a Boolean integer (0 or 1), depending on if it is true or
        false that the system is dynamically stable at a certain parameter
        configuration

        Parameters
        ----------
        params : Dictionary
              System parameres

        Returns
    
        -------
        s: Integer
           0 if unstable, 1 if unstable

        """
        # Get the non-Hermitian Hamiltonian and diagonalize it
        H_nh = self.get_H_nh(params)
        eigvals = np.linalg.eigvals(H_nh)
        
        # Is there any eigenvalue with positive imaginary part?
        s = np.any(np.imag(eigvals)>0)
        
        # Turn into integer and return
        return int(s)
    
    def get_PBC_and_OBC_gaps(self, params):
        """
        When comparing the singular values of the non-Hermitian Hamiltonian,
        topological zero-states appear only in open boundary conditions (OBC)
        although they are not strictly zero due to finite size effects.
        
        Here, we compute the spectral size of th8is finite size effect 
        (Delta_OBC) vs the infinite system gap (Delta_PBC)
        """
        # Initialize output list
        eigvals_k = np.array([])

        # Momentum space
        for k in np.linspace(-np.pi, np.pi, self.N):
            H = self.get_doubled_Hamiltonian(self.get_H_nh_k_space(k, params))
            eig = np.linalg.eigvalsh(H)
            eigvals_k = np.append(eigvals_k, eig)
    
        # Real space
        H = self.get_doubled_Hamiltonian(self.get_H_nh(params))
        eigvals = np.linalg.eigvalsh(H)
        
        # Get Delta_PBC
        Delta_PBC = np.min(np.abs(eigvals_k))
        
        # Topological modes emerge in OBC, with an energy below Delta_OBC
        modes = np.abs(eigvals)
        topo_modes = modes[modes<Delta_PBC]
        
        if len(topo_modes) == 0: Delta_OBC = 1e-10
        else: Delta_OBC = np.max(topo_modes)
        
        return Delta_PBC, Delta_OBC
        
        
   
        
    
    
    
    
    
    
    




        
        
        
        
    
   
     
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        