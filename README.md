# Topological-multi-mode-amplification-in-chiral-waveguides

In this repository, we present the code developed for the numerical simulations presented in the paper "Topological, multi-mode amplification induced by non-reciprocal, long-range dissipative couplings", by Carlos Vega, Alberto Muñoz de las Heras, Diego Porras and Alejandro González Tudela. The repository contains the following files:

1. GaussianSystem.py : In this file, we build a Python class that implements several methods regarding bosonic Gaussian systems, applied to a lattice of cavity modes. These modes are coupled through a chiral multi-mode waveguide, that is adiabatically eliminated, leading to effective chiral and long-range hoppings between the cavities.

2. Topological_Phase_Diagrams.ipynb: Here, we compute the driven-dissipative topological phase diagram considering two main ways of driving the system: an incoherent pump and a local parametric driving.

3. Topological_Amplification.ipynb: In this notebook, we explore the phenomenon of directional amplification that occurs when the topological invariant is non-zero. In particular, we explore the effects of multi-mode topological amplification.

4. Dynamics_and_Metastability.ipynb: The existence of multiple amplification channels is connected to the emergence of a metastable subspace, that becomes stable in the limit of an infinite lattice. In this notebook, we explore this phenomenon, as well as the time evolution of the distribution of cavity coherences.
