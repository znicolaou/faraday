#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
import matplotlib
import matplotlib.pyplot as plt

def viscid_mat (omega, kx, ky):
    kappax = kx + args.k1x * np.arange(-args.N,args.N+1)[:,np.newaxis] + args.k2x * np.arange(-args.N,args.N+1)[np.newaxis,:]
    kappay = ky + args.k1y * np.arange(-args.N,args.N+1)[:,np.newaxis] + args.k2y * np.arange(-args.N,args.N+1)[np.newaxis,:]
    Omega = 1j*(omega+2*np.pi*args.freq*np.arange(-args.N,args.N+1))[:,np.newaxis,np.newaxis]+args.mu/args.rho*(kappax**2 + kappay**2)[np.newaxis,:,:]
    return kappax,kappay,Omega

def inviscid_mat (omega, kx, ky):
    kappax = kx + args.k1x * np.arange(-args.N,args.N+1)[:,np.newaxis] + args.k2x * np.arange(-args.N,args.N+1)[np.newaxis,:]
    kappay = ky + args.k1y * np.arange(-args.N,args.N+1)[:,np.newaxis] + args.k2y * np.arange(-args.N,args.N+1)[np.newaxis,:]
    return kappax,kappay

if __name__ == "__main__":

    #Command line arguments
    parser = argparkappax,kappay,Omegase.ArgumentParser(description='Find the Floquet exponents for a given drive and wavenumber.')
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

    parser.add_argument("--N", type=int, required=False, default=5, dest='N', help='Number of modes to include the Floquet expansion')

    args = parser.parse_args()

    kappax,kappay,Omega=viscid_mat(1,args.kx,args.ky)
    print(kappax.shape,kappay.shape,Omega.shape)
