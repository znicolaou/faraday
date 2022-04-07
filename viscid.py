#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
from scipy.linalg import eig

#Return a numpy array corresponding to E^{klmn}_{k'l'm'n'} in Eqs. 31-39 in the notes. Return array has shape (3,3,2*Nt-1,2*Nt-1,2*Nx-1,2*Nx-1,2*Ny-1,2*Ny-1), with axes correponding to (k,k',l,l',m,m',n,n').
def viscid_mat (omega, kx, ky):
    #return array
    E = np.array((3,3,2*args.Nt-1,2*args.Nt-1,2*args.Nx-1,2*args.Nx-1,2*args.Ny-1,2*args.Ny-1))
    #mode indices. first axis for l, second for m, third for n
    ls=np.arange(-args.Nt,args.Nt+1)[:,np.newaxis,np.newaxis]
    ms=np.arange(-args.Nx,args.Nx+1)[np.newaxis,:,np.newaxis]
    ns=np.arange(-args.Ny,args.Ny+1)[np.newaxis,np.newaxis,:]

    kappax = kx + args.k1x*ms + args.k2x*ns
    kappay = ky + args.k1y*ms + args.k2y*ns
    Omega = 1j*(omega + 2*np.pi*args.freq*ls + args.mu/args.rho*(kappax**2 + kappay**2))
    #Define also here the symmetric sums and differences C,Ctile,S,Stilde above Eq. 31, and then assign the corresponding elements of E with array slicing.
    return E

#Return numpy arrays corresponding to C^{mn}_{m'n'} and D^{mn}_{m'n'} in Eqs. 41-42 in the notes. Return array has shape (2*Nx-1,2*Nx-1,2*Ny-1,2*Ny-1), with axes correponding to (m,m',n,n').
def inviscid_mat (omega, kx, ky):
    #return arrays
    F = np.array((2*args.Nx-1,2*args.Nx-1,2*args.Ny-1,2*args.Ny-1))
    G = np.array((2*args.Nx-1,2*args.Nx-1,2*args.Ny-1,2*args.Ny-1))
    #mode indices. first axis for m, second for n.
    ms=np.arange(-args.Nx,args.Nx+1)[:,np.newaxis]
    ns=np.arange(-args.Ny,args.Ny+1)[np.newaxis,:]
    kappax = kx + args.k1x*ms + args.k2x*ns
    kappay = ky + args.k1y*ms + args.k2y*ns
    #Define also here the symmetric sums and differences C,S above Eq. 31, and then assign the corresponding elements of F,G with array slicing.


    return F,G

#include this in case we want to import functions here elsewhere, in a jupyter notebook for example.
if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Find the Floquet exponents for a given drive and wavenumber.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--frequency", type=float, required=False, default=20, dest='freq', help='Driving frequency (Hz)')
    parser.add_argument("--acceleration", type=float, required=False, default=0.1, dest='ad', help='Driving acceleration (in gravitational units)')
    parser.add_argument("--viscosity", type=float, required=False, default=0.01, dest='mu', help='Viscosity (cgs units)')
    parser.add_argument("--density", type=float, required=False, default=1.0, dest='rho', help='Fluid density (cgs units)')
    parser.add_argument("--gravity", type=float, required=False, default=980, dest='g', help='Gravitational acceleration (cgs units)')
    parser.add_argument("--tension", type=float, required=False, default=72, dest='sigma', help='Sargparseurface tension (cgs units)')
    parser.add_argument("--kx", type=float, required=False, default=1, dest='kx', help='Wave vector x component')
    parser.add_argument("--ky", type=float, required=False, default=0, dest='ky', help='Wave vector y component')
    parser.add_argument("--height", type=float, required=False, default=0, dest='h0', help='Fluid depth')
    parser.add_argument("-kappax-as", type=float, required=False, default=0, dest='as', help='Substrate height')
    parser.add_argument("--k1x", type=float, required=False, default=1, dest='k1x', help='First reciprocal lattice vector x component')
    parser.add_argument("--k1y", type=float, required=False, default=0, dest='k1y', help='First reciprocal lattice vector y component')
    parser.add_argument("--k2x", type=float, required=False, default=0, dest='k2x', help='Second reciprocal lattice vector x component')
    parser.add_argument("--k2y", type=float, required=False, default=1, dest='k2y', help='Second reciprocal lattice vector y component')
    parser.add_argument("--Nt", type=int, required=False, default=3, dest='Nt', help='Number of modes to include the Floquet expansion for time')
    parser.add_argument("--Nx", type=int, required=False, default=3, dest='Nx', help='Number of modes to include the Floquet expansion for spatial x')
    parser.add_argument("--Ny", type=int, required=False, default=3, dest='Ny', help='Number of modes to include the Floquet expansion for spatial y')
    args = parser.parse_args()

    #Find the inviscid undriven problem, and its eigenvalues and vectors
    F,G=viscid_mat(args.kx,args.ky)
    #save the matrices to with the filebase prefix
    F.tofile(args.filebase+'F.npy')
    G.tofile(args.filebase+'G.npy')
    evals,levecs,revecs= eig(F, G, left=True, right=True)

    #find the matrices for the eigenvalue problems, starting using the inviscid eigenvalue for the frequency
    omega=evals[-1]**0.5
    E=viscid_mat(omega,args.kx,args.ky)

    #save the matrices to with the filebase prefix
    E.tofile(args.filebase+'E.npy')
    #Later, iteratively adjust omega in the nonlinear continuation
