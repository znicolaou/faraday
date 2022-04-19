#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
from scipy.linalg import eig
from scipy.special import iv

#Return a numpy array corresponding to E^{klmn}_{k'l'm'n'} in Eqs. 31-39 in the notes. Return array has shape (3,3,2*Nt+1,2*Nt+1,2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (k',k,l',l,m',m,n',n).
def viscid_mat (omega, kx, ky):
    # return array
    Ftilde,Gtilde=viscid_mat2 (omega, kx, ky)
    E=Ftilde-args.ad*Gtilde
    return E

def viscid_mat2 (omega, kx, ky):
    # return array
    Ftilde = np.zeros((3, 3, 2*args.Nt+1, 2*args.Nt+1, 2*args.Nx+1, 2*args.Nx+1, 2*args.Ny+1, 2*args.Ny+1),dtype=np.complex128)
    Gtilde = np.zeros((3, 3, 2*args.Nt+1, 2*args.Nt+1, 2*args.Nx+1, 2*args.Nx+1, 2*args.Ny+1, 2*args.Ny+1),dtype=np.complex128)

    # mode indices. axes appear in the order (l',l,m',m,n',n)
    lps = np.arange(-args.Nt, args.Nt + 1)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    ls = np.arange(-args.Nt, args.Nt + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    mps = np.arange(-args.Nx, args.Nx + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    ms = np.arange(-args.Nx, args.Nx + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    nps = np.arange(-args.Ny, args.Ny + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    ns = np.arange(-args.Ny, args.Ny + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    kappax = kx + args.k1x*ms + args.k2x*ns
    kappay = ky + args.k1y*ms + args.k2y*ns
    kappa = (kappax**2+kappay**2)**0.5
    Omega = 1j*(omega + 2*np.pi*args.freq*ls) + args.mu/args.rho*kappa**2

    # Define also here the symmetric sums and differences C, Ctilde, S, Stilde above Eq. 31, and then assign the
    # corresponding elements of E with array slicing.
    C = np.exp(-kappa*args.h0) * iv(ms-mps, kappa*args.As*0.5) * iv(ns-nps, kappa*args.As*0.5) + \
        np.exp(kappa*args.h0) * iv(ms-mps, -kappa*args.As*0.5) * iv(ns-nps, -kappa*args.As*0.5)

    Ctilde = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * iv(ms-mps, (args.rho/args.mu*Omega)**0.5*args.As*0.5) * \
             iv(ns-nps, (args.rho/args.mu*Omega)**0.5*args.As*0.5) + np.exp((args.rho/args.mu*Omega)**0.5*args.h0) * \
             iv(ms-mps, -(args.rho/args.mu*Omega)**0.5*args.As*0.5) * iv(ns-nps, -(args.rho/args.mu*Omega)**0.5*args.As*0.5)

    S = - np.exp(-kappa*args.h0) * iv(ms-mps, kappa*args.As*0.5) * iv(ns-nps, kappa*args.As*0.5) + \
        np.exp(kappa*args.h0) * iv(ms-mps, -kappa*args.As*0.5) * iv(ns-nps, -kappa*args.As*0.5)

    Stilde = - np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * iv(ms-mps, (args.rho/args.mu*Omega)**0.5*args.As*0.5) * \
             iv(ns-nps, (args.rho/args.mu*Omega)**0.5*args.As*0.5) + np.exp((args.rho/args.mu*Omega)**0.5*args.h0) * \
             iv(ms-mps, -(args.rho/args.mu*Omega)**0.5*args.As*0.5) * iv(ns-nps, -(args.rho/args.mu*Omega)**0.5*args.As*0.5)

    Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        (Ctilde - (2*args.mu*kappax**2*C)/(args.rho*Omega+args.mu*kappa**2))

    Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        2*args.mu*kappax*kappay*C/(args.rho*Omega+args.mu*kappa**2)

    Ftilde[2, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*args.mu*(args.rho/args.mu*Omega)**0.5 * \
        (Ctilde-Stilde) + (args.rho*Omega+args.mu*kappa**2)*S/kappa + (args.g*args.rho**2+kappa**2*(args.rho*args.sigma - \
        4*args.mu**2*(args.rho*Omega/args.mu)**0.5))*C/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

    Gtilde[2, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,1:]

    Gtilde[2, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,:-1]

    Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        2*args.mu*kappax*kappay*C/(args.rho*Omega+args.mu*kappa**2)

    Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        (Ctilde - (2*args.mu*kappay**2*C)/(args.rho*Omega+args.mu*kappa**2))

    Ftilde[2, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappay*(2*args.mu*(args.rho/args.mu*Omega)**0.5 * \
        (Ctilde-Stilde) + (args.rho*Omega+args.mu*kappa**2)*S/kappa + (args.g*args.rho**2+kappa**2*(args.rho*args.sigma - \
        4*args.mu**2*(args.rho*Omega/args.mu)**0.5))*C/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

    Gtilde[2, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*args.rho*kappay*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,1:]

    Gtilde[2, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*args.rho*kappay*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,:-1]

    Ftilde[0, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        kappax*(-2*args.mu*kappa*S/(args.rho*Omega+args.mu*kappa**2)+Stilde/(args.rho*Omega/args.mu)**0.5)

    Ftilde[1, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        kappay*(-2*args.mu*kappa*S/(args.rho*Omega+args.mu*kappa**2)+Stilde/(args.rho*Omega/args.mu)**0.5)

    Ftilde[2, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*args.mu*kappa**2*(Ctilde-Stilde)+(args.rho*Omega + \
        args.mu*kappa**2)*C+kappa*(args.g*args.rho**2+kappa**2*(args.rho*args.sigma-4*args.mu**2*(args.rho * \
        Omega/args.mu)**0.5))*S/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

    Gtilde[2, 2][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                      args.mu*kappa**2))[:,1:]

    Gtilde[2, 2][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                      args.mu*kappa**2))[:,:-1]

    return Ftilde,Gtilde


def viscid_flat_mat (omega, kx, ky):
    # return array
    Ftilde = np.zeros((3, 3, 2*args.Nt+1, 2*args.Nt+1),dtype=np.complex128)
    Gtilde = np.zeros((3, 3, 2*args.Nt+1, 2*args.Nt+1),dtype=np.complex128)

    # mode indices. axes appear in the order (l',l,m',m,n',n)
    lps = np.arange(-args.Nt, args.Nt + 1)[:, np.newaxis]
    ls = np.arange(-args.Nt, args.Nt + 1)[np.newaxis, :]

    kappax = kx
    kappay = ky
    kappa = (kappax**2+kappay**2)**0.5
    Omega = 1j*(omega + 2*np.pi*args.freq*ls) + args.mu/args.rho*kappa**2

    # Define also here the symmetric sums and differences C, Ctilde, S, Stilde above Eq. 31, and then assign the
    # corresponding elements of E with array slicing.
    C = np.exp(-kappa*args.h0) + np.exp(kappa*args.h0)

    Ctilde = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0)  + np.exp((args.rho/args.mu*Omega)**0.5*args.h0)

    S = - np.exp(-kappa*args.h0)  + np.exp(kappa*args.h0)

    Stilde = - np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) + np.exp((args.rho/args.mu*Omega)**0.5*args.h0)

    Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        (Ctilde - (2*args.mu*kappax**2*C)/(args.rho*Omega+args.mu*kappa**2))

    Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        2*args.mu*kappax*kappay*C/(args.rho*Omega+args.mu*kappa**2)

    Ftilde[2, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*args.mu*(args.rho/args.mu*Omega)**0.5 * \
        (Ctilde-Stilde) + (args.rho*Omega+args.mu*kappa**2)*S/kappa + (args.g*args.rho**2+kappa**2*(args.rho*args.sigma - \
        4*args.mu**2*(args.rho*Omega/args.mu)**0.5))*C/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

    Gtilde[2, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,1:]

    Gtilde[2, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,:-1]

    Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        2*args.mu*kappax*kappay*C/(args.rho*Omega+args.mu*kappa**2)

    Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        (Ctilde - (2*args.mu*kappay**2*C)/(args.rho*Omega+args.mu*kappa**2))

    Ftilde[2, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappay*(2*args.mu*(args.rho/args.mu*Omega)**0.5 * \
        (Ctilde-Stilde) + (args.rho*Omega+args.mu*kappa**2)*S/kappa + (args.g*args.rho**2+kappa**2*(args.rho*args.sigma - \
        4*args.mu**2*(args.rho*Omega/args.mu)**0.5))*C/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

    Gtilde[2, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*args.rho*kappay*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,1:]

    Gtilde[2, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*args.rho*kappay*C/(4*(args.rho*Omega +
                                                                                                     args.mu*kappa**2))[:,:-1]

    Ftilde[0, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        kappax*(-2*args.mu*kappa*S/(args.rho*Omega+args.mu*kappa**2)+Stilde/(args.rho*Omega/args.mu)**0.5)

    Ftilde[1, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
        kappay*(-2*args.mu*kappa*S/(args.rho*Omega+args.mu*kappa**2)+Stilde/(args.rho*Omega/args.mu)**0.5)

    Ftilde[2, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*args.mu*kappa**2*(Ctilde-Stilde)+(args.rho*Omega + \
        args.mu*kappa**2)*C+kappa*(args.g*args.rho**2+kappa**2*(args.rho*args.sigma-4*args.mu**2*(args.rho * \
        Omega/args.mu)**0.5))*S/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

    Gtilde[2, 2][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                      args.mu*kappa**2))[:,1:]

    Gtilde[2, 2][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                      args.mu*kappa**2))[:,:-1]

    return Ftilde,Gtilde


#Return numpy arrays corresponding to C^{mn}_{m'n'} and D^{mn}_{m'n'} in Eqs. 41-42 in the notes. Return array has shape (2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (m',m,n',n).
def inviscid_mat (kx, ky):
    #mode indices. axes appear in the order (l',l,m',m,n',n)
    mps=np.arange(-args.Nx,args.Nx+1)[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    ms=np.arange(-args.Nx,args.Nx+1)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
    nps=np.arange(-args.Ny,args.Ny+1)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    ns=np.arange(-args.Ny,args.Ny+1)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

    kappax = kx + args.k1x*ms + args.k2x*ns
    kappay = ky + args.k1y*ms + args.k2y*ns
    kappa = (kappax**2+kappay**2)**0.5

    C = np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As/2)*iv(ns-nps,kappa*args.As/2) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As/2)*iv(ns-nps,-kappa*args.As/2)
    S = -np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As/2)*iv(ns-nps,kappa*args.As/2) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As/2)*iv(ns-nps,-kappa*args.As/2)

    F=kappa*(args.g+args.sigma/args.rho*kappa**2)*(1-((ns-nps)*(args.k1x*kappax+args.k1y*kappay)+(ms-mps)*(args.k2x*kappax+args.k2y*kappay))/kappa**2)*S
    G=-(1-((ns-nps)*(args.k1x*kappax+args.k1y*kappay)+(ms-mps)*(args.k2x*kappax+args.k2y*kappay))/kappa**2)*C

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
    parser.add_argument("--tension", type=float, required=False, default=72, dest='sigma', help='Surface tension (cgs units)')
    parser.add_argument("--kx", type=float, required=False, default=1, dest='kx', help='Wave vector x component')
    parser.add_argument("--ky", type=float, required=False, default=0, dest='ky', help='Wave vector y component')
    parser.add_argument("--height", type=float, required=False, default=0, dest='h0', help='Fluid depth')
    parser.add_argument("--as", type=float, required=False, default=0, dest='As', help='Substrate height')
    parser.add_argument("--k1x", type=float, required=False, default=1, dest='k1x', help='First reciprocal lattice vector x component')
    parser.add_argument("--k1y", type=float, required=False, default=0, dest='k1y', help='First reciprocal lattice vector y component')
    parser.add_argument("--k2x", type=float, required=False, default=0, dest='k2x', help='Second reciprocal lattice vector x component')
    parser.add_argument("--k2y", type=float, required=False, default=1, dest='k2y', help='Second reciprocal lattice vector y component')
    parser.add_argument("--Nt", type=int, required=False, default=3, dest='Nt', help='Number of modes to include the Floquet expansion for time')
    parser.add_argument("--Nx", type=int, required=False, default=3, dest='Nx', help='Number of modes to include the Floquet expansion for spatial x')
    parser.add_argument("--Ny", type=int, required=False, default=3, dest='Ny', help='Number of modes to include the Floquet expansion for spatial y')
    args = parser.parse_args()

    #Find the inviscid undriven problem, and its eigenvalues and vectors
    F,G=inviscid_mat(args.kx,args.ky)
    Fflattened=np.transpose(F[0,0],(0,2,1,3)).reshape(((2*args.Nx+1)*(2*args.Ny+1),(2*args.Nx+1)*(2*args.Ny+1)))
    Gflattened=np.transpose(G[0,0],(0,2,1,3)).reshape(((2*args.Nx+1)*(2*args.Ny+1),(2*args.Nx+1)*(2*args.Ny+1)))
    evals,evecs=eig(Fflattened,Gflattened)

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

else:
    #For benchmarking in jupyter
    class args:
        Nx=5
        Ny=5
        Nt=5
        h0=0.1
        As=0.05
        k1x=3**0.5/2*np.pi
        k1y=-0.5*np.pi
        k2x=0
        k2y=np.pi
        freq=5
        g=980
        sigma=20
        rho=1
        mu=0.005
        ad=0.1
