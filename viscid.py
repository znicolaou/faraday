#!/usr/bin/env python
import numpy as np
import argparse
import sys
import os
import json
from scipy.linalg import eig
from scipy.special import iv
from scipy.linalg import solve
from scipy.linalg import LinAlgWarning
import warnings
warnings.filterwarnings("ignore",category=LinAlgWarning)
# warnings.filterwarnings("error",category=LinAlgWarning)


#Return a numpy array corresponding to E^{klmn}_{k'l'm'n'} in Eqs. 31-39 in the notes. Return array has shape (3,3,2*Nt+1,2*Nt+1,2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (k',k,l',l,m',m,n',n).
def viscid_mat (omega, args):
    Ftilde,Gtilde=viscid_boundary (omega, args)
    E=Ftilde-args.ad*Gtilde
    return E

#Return numpy arrays corresponding to Ftilde^{klmn}_{k'l'm'n'} and Gtilde^{klmn}_{k'l'm'n'} in Eqs. 54 in the notes.
def viscid_boundary (omega, args):
    if args.dim==1:
        # return array
        Ftilde = np.zeros((2, 2, 2*args.Nt+1, 2*args.Nt+1, 2*args.Nx+1, 2*args.Nx+1),dtype=np.complex128)
        Gtilde = np.zeros((2, 2, 2*args.Nt+1, 2*args.Nt+1, 2*args.Nx+1, 2*args.Nx+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-args.Nt, args.Nt + 1)[:, np.newaxis, np.newaxis, np.newaxis]
        ls = np.arange(-args.Nt, args.Nt + 1)[np.newaxis, :, np.newaxis, np.newaxis]
        mps = np.arange(-args.Nx, args.Nx + 1)[np.newaxis, np.newaxis, :, np.newaxis]
        ms = np.arange(-args.Nx, args.Nx + 1)[np.newaxis, np.newaxis, np.newaxis, :]

        kappax = args.kx + args.k1x*ms
        kappa = np.abs(kappax)
        Omega = 1j*(omega + 2*np.pi*args.freq*ls) + args.mu/args.rho*kappa**2

        C = np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As)

        Ctilde = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * iv(ms-mps, (args.rho/args.mu*Omega)**0.5*args.As) + np.exp((args.rho/args.mu*Omega)**0.5*args.h0) * iv(ms-mps, -(args.rho/args.mu*Omega)**0.5*args.As)

        S = -np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As)

        Stilde = - np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * iv(ms-mps, (args.rho/args.mu*Omega)**0.5*args.As) + np.exp((args.rho/args.mu*Omega)**0.5*args.h0) * iv(ms-mps, -(args.rho/args.mu*Omega)**0.5*args.As)

        Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
            (Ctilde - (2*args.mu*kappax**2*C)/(args.rho*Omega+args.mu*kappa**2))

        Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*args.mu*(args.rho/args.mu*Omega)**0.5 * \
            (Ctilde-Stilde) + (args.rho*Omega+args.mu*kappa**2)*S/kappa + (args.g*args.rho**2+kappa**2*(args.rho*args.sigma - \
            4*args.mu**2*(args.rho*Omega/args.mu)**0.5))*C/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

        Gtilde[1, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                         args.mu*kappa**2))[:,1:]

        Gtilde[1, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                         args.mu*kappa**2))[:,:-1]

        Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
            kappax*(-2*args.mu*kappa*S/(args.rho*Omega+args.mu*kappa**2)+Stilde/(args.rho*Omega/args.mu)**0.5)

        Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*args.mu*kappa**2*(Ctilde-Stilde)+(args.rho*Omega + \
            args.mu*kappa**2)*C+kappa*(args.g*args.rho**2+kappa**2*(args.rho*args.sigma-4*args.mu**2*(args.rho * \
            Omega/args.mu)**0.5))*S/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

        Gtilde[1, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                          args.mu*kappa**2))[:,1:]

        Gtilde[1, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                          args.mu*kappa**2))[:,:-1]

        return Ftilde,Gtilde

    elif args.dim==2:
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

        kappax = args.kx + args.k1x*ms + args.k2x*ns
        kappay = args.ky + args.k1y*ms + args.k2y*ns
        kappa = (kappax**2+kappay**2)**0.5
        Omega = 1j*(omega + 2*np.pi*args.freq*ls) + args.mu/args.rho*kappa**2

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

#Return numpy arrays corresponding to C^{mn}_{m'n'} and D^{mn}_{m'n'} in Eqs. 41-42 in the notes. Return array has shape (2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (m',m,n',n).
def inviscid_mat (args):
    if args.dim==1:
        #mode indices. axes appear in the order (l',l,m',m)
        mps=np.arange(-args.Nx,args.Nx+1)[np.newaxis,np.newaxis,:,np.newaxis]
        ms=np.arange(-args.Nx,args.Nx+1)[np.newaxis,np.newaxis,np.newaxis,:]

        kappax = args.kx + args.k1x*ms
        kappa = np.abs(kappax)
        C = np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As)
        S = -np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As)

        F=kappa*(args.g+args.sigma/args.rho*kappa**2)*(1-((ms-mps)*(args.k1x*kappax))/kappa**2)*S
        G=-(1-((ms-mps)*(args.k1x*kappax))/kappa**2)*C

    elif args.dim==2:
        #mode indices. axes appear in the order (l',l,m',m,n',n)
        mps=np.arange(-args.Nx,args.Nx+1)[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        ms=np.arange(-args.Nx,args.Nx+1)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
        nps=np.arange(-args.Ny,args.Ny+1)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
        ns=np.arange(-args.Ny,args.Ny+1)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

        kappax = args.kx + args.k1x*ms + args.k2x*ns
        kappay = args.ky + args.k1y*ms + args.k2y*ns
        kappa = (kappax**2+kappay**2)**0.5

        C = np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As/2)*iv(ns-nps,kappa*args.As/2) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As/2)*iv(ns-nps,-kappa*args.As/2)
        S = -np.exp(-kappa*args.h0)*iv(ms-mps,kappa*args.As/2)*iv(ns-nps,kappa*args.As/2) + np.exp(kappa*args.h0)*iv(ms-mps,-kappa*args.As/2)*iv(ns-nps,-kappa*args.As/2)

        F=kappa*(args.g+args.sigma/args.rho*kappa**2)*(1-((ns-nps)*(args.k1x*kappax+args.k1y*kappay)+(ms-mps)*(args.k2x*kappax+args.k2y*kappay))/kappa**2)*S
        G=-(1-((ns-nps)*(args.k1x*kappax+args.k1y*kappay)+(ms-mps)*(args.k2x*kappax+args.k2y*kappay))/kappa**2)*C

    return F,G

def viscid_flat_boundary (omega, args):
    if args.dim==1:
        # return array
        Ftilde = np.zeros((2, 2, 2*args.Nt+1, 2*args.Nt+1),dtype=np.complex128)
        Gtilde = np.zeros((2, 2, 2*args.Nt+1, 2*args.Nt+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-args.Nt, args.Nt + 1)[:, np.newaxis]
        ls = np.arange(-args.Nt, args.Nt + 1)[np.newaxis, :]

        kappax = args.kx
        kappa = np.abs(kappax)
        Omega = 1j*(omega + 2*np.pi*args.freq*ls) + args.mu/args.rho*kappa**2
        C = np.exp(-kappa*args.h0) + np.exp(kappa*args.h0)

        Ctilde = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0)  + np.exp((args.rho/args.mu*Omega)**0.5*args.h0)

        S = - np.exp(-kappa*args.h0)  + np.exp(kappa*args.h0)

        Stilde = - np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) + np.exp((args.rho/args.mu*Omega)**0.5*args.h0)

        Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
            (Ctilde - (2*args.mu*kappax**2*C)/(args.rho*Omega+args.mu*kappa**2))

        Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*args.mu*(args.rho/args.mu*Omega)**0.5 * \
            (Ctilde-Stilde) + (args.rho*Omega+args.mu*kappa**2)*S/kappa + (args.g*args.rho**2+kappa**2*(args.rho*args.sigma - \
            4*args.mu**2*(args.rho*Omega/args.mu)**0.5))*C/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

        Gtilde[1, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                         args.mu*kappa**2))[:,1:]

        Gtilde[1, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*args.rho*kappax*C/(4*(args.rho*Omega +
                                                                                                         args.mu*kappa**2))[:,:-1]

        Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(args.rho/args.mu*Omega)**0.5*args.h0) * \
            kappax*(-2*args.mu*kappa*S/(args.rho*Omega+args.mu*kappa**2)+Stilde/(args.rho*Omega/args.mu)**0.5)

        Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*args.mu*kappa**2*(Ctilde-Stilde)+(args.rho*Omega + \
            args.mu*kappa**2)*C+kappa*(args.g*args.rho**2+kappa**2*(args.rho*args.sigma-4*args.mu**2*(args.rho * \
            Omega/args.mu)**0.5))*S/(args.rho*Omega+args.mu*kappa**2))/(2*args.rho)

        Gtilde[1, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                          args.mu*kappa**2))[:,1:]

        Gtilde[1, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = args.rho*kappa*S/(4*(args.rho*Omega + \
                                                                                                          args.mu*kappa**2))[:,:-1]

        return Ftilde,Gtilde

    elif args.dim==2:
        # return array
        Ftilde = np.zeros((3, 3, 2*args.Nt+1, 2*args.Nt+1),dtype=np.complex128)
        Gtilde = np.zeros((3, 3, 2*args.Nt+1, 2*args.Nt+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-args.Nt, args.Nt + 1)[:, np.newaxis]
        ls = np.arange(-args.Nt, args.Nt + 1)[np.newaxis, :]

        kappax = args.kx
        kappay = args.ky
        kappa = (kappax**2+kappay**2)**0.5
        Omega = 1j*(omega + 2*np.pi*args.freq*ls) + args.mu/args.rho*kappa**2
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

#Rayleigh quotient iteration for the viscid floquet problem
def rayleigh(omega_0, v0, w0, args):
    vn=v0
    wn=w0

    omegas=[]
    vns=[]
    wns=[]
    omega=omega_0
    for n in range(args.itmax):
        E_n=viscid_mat(omega, args)
        dE=(viscid_mat(omega+args.domega_fd,args)-E_n)/args.domega_fd
        if args.dim==1:
            flat=np.transpose(E_n,(0,2,4,1,3,5)).reshape((2*(2*args.Nt+1)*(2*args.Nx+1),2*(2*args.Nt+1)*(2*args.Nx+1)))
            bflat=np.einsum("kKlLmM,KLM",dE,vn).reshape(2*(2*args.Nt+1)*(2*args.Nx+1))
            btflat=np.einsum("kKlLmM,klm",dE,wn).reshape(2*(2*args.Nt+1)*(2*args.Nx+1))
        if args.dim==2:
            flat=np.transpose(E_n,(0,2,4,6,1,3,5,7)).reshape((3*(2*args.Nt+1)*(2*args.Nx+1)*(2*args.Ny+1),3*(2*args.Nt+1)*(2*args.Nx+1)*(2*args.Ny+1)))
            bflat=np.einsum("kKlLmMnN,KLMN",dE,vn).reshape(3*(2*args.Nt+1)*(2*args.Nx+1)*(2*args.Ny+1))
            btflat=np.einsum("kKlLmMnN,klmn",dE,wn).reshape(3*(2*args.Nt+1)*(2*args.Nx+1)*(2*args.Ny+1))
        try:
            xi=solve(flat, bflat)
            zeta=solve(flat.T, btflat)
            if args.dim==1:
                xi=xi.reshape(2,(2*args.Nt+1),(2*args.Nx+1))
                zeta=zeta.reshape(2,(2*args.Nt+1),(2*args.Nx+1))
                domega=-np.einsum("kKlLmM,KLM,klm",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn)
            if args.dim==2:
                xi=xi.reshape(3,(2*args.Nt+1),(2*args.Nx+1),(2*args.Ny+1))
                zeta=zeta.reshape(3,(2*args.Nt+1),(2*args.Nx+1),(2*args.Ny+1))
                domega=-np.einsum("kKlLmMnN,KLMN,klmn",E_n,vn,wn)/np.einsum("kKlLmMnN,KLMN,klmn",dE,vn,wn)

            omega=omega+domega
            vn=(xi/np.linalg.norm(xi))
            wn=zeta/np.linalg.norm(zeta)
            omegas=omegas+[omega]
            vns=vns+[vn]
            wns=wns+[wn]
        except LinAlgWarning:
            if args.dim==1:
                domega=-np.einsum("kKlLmM,KLM,klm",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn)
            if args.dim==2:
                domega=-np.einsum("kKlLmMnN,KLMN,klmn",E_n,vn,wn)/np.einsum("kKlLmMnN,KLMN,klmn",dE,vn,wn)

            omega=omega+domega
            omegas=omegas+[omega]
            vns=vns+[vn]
            wns=wns+[wn]
            break

    if args.dim==1:
        print(n, omega, np.einsum("kKlLmM,KLM,klm",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn))
    if args.dim==2:
        print(n, omega, np.einsum("kKlLmMnN,KLMN,klmn",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn))
    # return omegas[-1],vns[-1],wns[-1]
    return omegas,vns,wns

#include this in case we want to import functions here elsewhere, in a jupyter notebook for example.
if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Find the Floquet exponents for a given drive and wavenumber.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--frequency", type=float, required=False, default=5, dest='freq', help='Driving frequency (Hz)')
    parser.add_argument("--acceleration", type=float, required=False, default=0.0, dest='ad', help='Driving acceleration (in gravitational units)')
    parser.add_argument("--dacc", type=float, required=False, default=0.5, dest='dad', help='Driving acceleration steps (in gravitational units)')
    parser.add_argument("--viscosity", type=float, required=False, default=0.005, dest='mu', help='Viscosity (cgs units)')
    parser.add_argument("--density", type=float, required=False, default=1.0, dest='rho', help='Fluid density (cgs units)')
    parser.add_argument("--gravity", type=float, required=False, default=980, dest='g', help='Gravitational acceleration (cgs units)')
    parser.add_argument("--tension", type=float, required=False, default=20, dest='sigma', help='Surface tension (cgs units)')
    parser.add_argument("--kx", type=float, required=False, default=0.5*np.pi, dest='kx', help='Wave vector x component')
    parser.add_argument("--ky", type=float, required=False, default=0, dest='ky', help='Wave vector y component')
    parser.add_argument("--height", type=float, required=False, default=0.1, dest='h0', help='Fluid depth')
    parser.add_argument("--as", type=float, required=False, default=0.05, dest='As', help='Substrate height')
    parser.add_argument("--k1x", type=float, required=False, default=np.pi, dest='k1x', help='Second reciprocal lattice vector x component')
    parser.add_argument("--k1y", type=float, required=False, default=0, dest='k1y', help='Second reciprocal lattice vector y component')
    parser.add_argument("--k2x", type=float, required=False, default=-0.5*np.pi, dest='k2x', help='First reciprocal lattice vector x component')
    parser.add_argument("--k2y", type=float, required=False, default=3**0.5/2*np.pi, dest='k2y', help='First reciprocal lattice vector y component')
    parser.add_argument("--Nt", type=int, required=False, default=5, dest='Nt', help='Number of modes to include the Floquet expansion for time')
    parser.add_argument("--Nx", type=int, required=False, default=5, dest='Nx', help='Number of modes to include the Floquet expansion for spatial x')
    parser.add_argument("--Ny", type=int, required=False, default=5, dest='Ny', help='Number of modes to include the Floquet expansion for spatial y')
    parser.add_argument("--dim", type=int, required=False, default=2, dest='dim', help='Dimension, 1 or 2. Default 2.')
    parser.add_argument("--nmodes", type=int, required=False, default=2, dest='nmodes', help='Number of modes to track. Default 2.')
    parser.add_argument("--itmax", type=int, required=False, default=5, dest='itmax', help='Number of iterators in acceleration continuation.')
    parser.add_argument("--num", type=int, required=False, default=0, dest='num', help='Number of iterators in acceleration continuation.')
    parser.add_argument("--domega_fd", type=float, required=False, default=0.1, dest='domega_fd', help='Finite difference step')
    parser.add_argument("--negatives", type=int, required=False, default=1, dest='negatives', help='Include negative frequencies')

    args = parser.parse_args()

    if not args.dim==1 and not args.dim==2:
        print('Dimension must be 1 or 2')
        exit(0)

    #If no initial modes exist, use the lowest frequency inviscid modes to start
    # if not os.path.isfile(args.filebase+'evals.npy') or not os.path.isfile(args.filebase+'evecs.npy'):
    print(args)
    json=json.dumps(args.__dict__)
    f=open(args.filebase+'args.json','w')
    f.write(json)
    f.close()
    As=args.As
    args.As=0
    F,G=inviscid_mat(args)
    if args.dim==1:
        Fflattened=F[0,0]
        Gflattened=G[0,0]
    elif args.dim==2:
        Fflattened=np.transpose(F[0,0],(0,2,1,3)).reshape(((2*args.Nx+1)*(2*args.Ny+1),(2*args.Nx+1)*(2*args.Ny+1)))
        Gflattened=np.transpose(G[0,0],(0,2,1,3)).reshape(((2*args.Nx+1)*(2*args.Ny+1),(2*args.Nx+1)*(2*args.Ny+1)))
    evals,levecs,revecs=eig(Fflattened,Gflattened,left=True)

    inds=np.argsort(-evals)[:args.nmodes]
    omegas=[]
    vns=[]
    wns=[]
    for i in range(args.nmodes):
        ind=inds[i]
        if args.dim==1:
            v0_inviscid=revecs[:,ind]
            # w0_inviscid=np.conjugate(levecs[:,ind])
            w0_inviscid=1
            v=np.zeros((2,(2*args.Nt+1),(2*args.Nx+1)),dtype=np.complex128)
            w=np.zeros((2,(2*args.Nt+1),(2*args.Nx+1)),dtype=np.complex128)

        elif args.dim==2:
            v0_inviscid=revecs[:,ind].reshape(((2*args.Nx+1),(2*args.Ny+1)))
            # w0_inviscid=np.conjugate(levecs[:,ind].reshape(((2*args.Nx+1),(2*args.Ny+1))))
            w0_inviscid=1
            v=np.zeros((3,(2*args.Nt+1),(2*args.Nx+1),(2*args.Ny+1)),dtype=np.complex128)
            w=np.zeros((3,(2*args.Nt+1),(2*args.Nx+1),(2*args.Ny+1)),dtype=np.complex128)

        omega_inviscid=(-evals[ind])**0.5
        print(omega_inviscid)
        if args.dim==1:
            v[1,args.Nt]=v0_inviscid
            w[1,args.Nt]=w0_inviscid
        if args.dim==2:
            v[2,args.Nt]=v0_inviscid
            w[2,args.Nt]=w0_inviscid


        #Should we iteratively increase as through num steps here??
        # omegas_i,vns_i,wns_i=rayleigh(omega_inviscid,v,w,args)
        omega=omega_inviscid
        args.As=0
        for it in range(args.num):
            args.As += As/args.num
            print(args.As)
            omegas_i,vns_i,wns_i=rayleigh(omega,v,w,args)
            omega=omegas_i[-1]
            v=vns_i[-1]
            w=wns_i[-1]
        print(args.As)
        omegas=omegas+[omegas_i[-1]]
        vns=vns+[vns_i[-1]]
        wns=wns+[wns_i[-1]]
    if args.negatives:
        for i in range(args.nmodes):
            ind=inds[i]
            if args.dim==1:
                v0_inviscid=revecs[:,ind]
                # w0_inviscid=np.conjugate(levecs[:,ind])
                w0_inviscid=1
                v=np.zeros((2,(2*args.Nt+1),(2*args.Nx+1)),dtype=np.complex128)
                w=np.zeros((2,(2*args.Nt+1),(2*args.Nx+1)),dtype=np.complex128)

            elif args.dim==2:
                v0_inviscid=revecs[:,ind].reshape(((2*args.Nx+1),(2*args.Ny+1)))
                # w0_inviscid=np.conjugate(levecs[:,ind].reshape(((2*args.Nx+1),(2*args.Ny+1))))
                w0_inviscid=1
                v=np.zeros((3,(2*args.Nt+1),(2*args.Nx+1),(2*args.Ny+1)),dtype=np.complex128)
                w=np.zeros((3,(2*args.Nt+1),(2*args.Nx+1),(2*args.Ny+1)),dtype=np.complex128)
            omega_inviscid=-(-evals[ind])**0.5
            print(omega_inviscid)
            if args.dim==1:
                v[1,args.Nt]=v0_inviscid
                w[1,args.Nt]=w0_inviscid
            if args.dim==2:
                v[2,args.Nt]=v0_inviscid
                w[2,args.Nt]=w0_inviscid

            args.As=0
            omega=omega_inviscid
            for it in range(args.num):
                args.As += As/args.num
                print(args.As)
                omegas_i,vns_i,wns_i=rayleigh(omega,v,w,args)
                omega=omegas_i[-1]
                v=vns_i[-1]
                w=wns_i[-1]
            omegas=omegas+[omegas_i[-1]]
            vns=vns+[vns_i[-1]]
            wns=wns+[wns_i[-1]]
    #save the matrices to with the filebase prefix
    np.save(args.filebase+"evals.npy",np.array(omegas))
    np.save(args.filebase+"evecs.npy",[np.array(vns),np.array(wns)])

    #Continuation along ad.
    ad0=args.ad
    omegas=np.load(args.filebase+'evals.npy')
    revecs,levecs=np.load(args.filebase+'evecs.npy')
    for i in range(len(omegas)):
        args.ad=ad0
        omegan=[omegas[i]]
        vn=[revecs[i]]
        wn=[levecs[i]]
        omegans=[omegan[-1]]
        vns=[vn[-1]]
        wns=[wn[-1]]

        for it in range(args.num):
            args.ad += args.g*args.dad/args.num
            print(args.ad)
            omegan,vn,wn=rayleigh(omegan[-1],vn[-1],wn[-1],args)
            omegans=omegans+[omegan[-1]]
            vns=vns+[vn[-1]]
            wns=wns+[wn[-1]]

        np.save(args.filebase+"evals_cont"+str(i)+".npy",np.array(omegans))
        np.save(args.filebase+"evecs_cont"+str(i)+".npy",[np.array(vns),np.array(wns)])

else:
    #For benchmarking in jupyter
    class args:
        Nx=5
        Ny=5
        Nt=5
        h0=0.1
        As=0.05
        k2y=3**0.5/2*np.pi
        k2x=-0.5*np.pi
        k1y=0
        k1x=np.pi
        freq=4
        g=980
        sigma=20
        rho=1
        mu=0.005
        ad=0.1
        domega_fd=0.1
        dim=2
        kx=0.5*np.pi
        ky=0
        def __str__(self):
            return str(self.__dict__)
