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
from scipy.linalg import LinAlgError
import warnings
warnings.filterwarnings("ignore",category=LinAlgWarning)
# warnings.filterwarnings("error",category=LinAlgWarning)


#Return a numpy array corresponding to E^{klmn}_{k'l'm'n'} in Eqs. 31-39 in the notes. Return array has shape (3,3,2*Nt+1,2*Nt+1,2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (k',k,l',l,m',m,n',n).
def viscid_mat (omega, argsdict):
    Ftilde,Gtilde=viscid_boundary (omega, argsdict)
    E=Ftilde-argsdict['ad']*Gtilde
    return E

#Return numpy arrays corresponding to Ftilde^{klmn}_{k'l'm'n'} and Gtilde^{klmn}_{k'l'm'n'} in Eqs. 54 in the notes.
def viscid_boundary (omega, argsdict):
    if argsdict['dim']==1:
        # return array
        Ftilde = np.zeros((2, 2, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1, 2*argsdict['Nx']+1, 2*argsdict['Nx']+1),dtype=np.complex128)
        Gtilde = np.zeros((2, 2, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1, 2*argsdict['Nx']+1, 2*argsdict['Nx']+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[:, np.newaxis, np.newaxis, np.newaxis]
        ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :, np.newaxis, np.newaxis]
        mps = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, :, np.newaxis]
        ms = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, np.newaxis, :]

        kappax = argsdict['kx'] + argsdict['k1x']*ms
        kappa = np.abs(kappax)
        Omega = 1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2

        C = np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As'])

        Ctilde = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']) + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As'])

        S = -np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As'])

        Stilde = - np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']) + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As'])

        Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))

        Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
            (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
            4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[1, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,1:]

        Gtilde[1, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,:-1]

        Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)

        Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
            argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
            Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[1, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,1:]

        Gtilde[1, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,:-1]

        return Ftilde,Gtilde

    elif argsdict['dim']==2:
        # return array
        Ftilde = np.zeros((3, 3, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1, 2*argsdict['Nx']+1, 2*argsdict['Nx']+1, 2*argsdict['Ny']+1, 2*argsdict['Ny']+1),dtype=np.complex128)
        Gtilde = np.zeros((3, 3, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1, 2*argsdict['Nx']+1, 2*argsdict['Nx']+1, 2*argsdict['Ny']+1, 2*argsdict['Ny']+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        mps = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        ms = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        nps = np.arange(-argsdict['Ny'], argsdict['Ny'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        ns = np.arange(-argsdict['Ny'], argsdict['Ny'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        kappax = argsdict['kx'] + argsdict['k1x']*ms + argsdict['k2x']*ns
        kappay = argsdict['ky'] + argsdict['k1y']*ms + argsdict['k2y']*ns
        kappa = (kappax**2+kappay**2)**0.5
        Omega = 1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2

        C = np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + \
            np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5)

        Ctilde = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * \
                 iv(ns-nps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
                 iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5)

        S = - np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + \
            np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5)

        Stilde = - np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * \
                 iv(ns-nps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
                 iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5)

        Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))

        Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            2*argsdict['mu']*kappax*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)

        Ftilde[2, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
            (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
            4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[2, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,1:]

        Gtilde[2, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,:-1]

        Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            2*argsdict['mu']*kappax*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)

        Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            (Ctilde - (2*argsdict['mu']*kappay**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))

        Ftilde[2, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappay*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
            (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
            4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[2, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*argsdict['rho']*kappay*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,1:]

        Gtilde[2, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*argsdict['rho']*kappay*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,:-1]

        Ftilde[0, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)

        Ftilde[1, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            kappay*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)

        Ftilde[2, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
            argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
            Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[2, 2][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,1:]

        Gtilde[2, 2][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,:-1]

        return Ftilde,Gtilde

#Return numpy arrays corresponding to C^{mn}_{m'n'} and D^{mn}_{m'n'} in Eqs. 41-42 in the notes. Return array has shape (2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (m',m,n',n).
def inviscid_mat (argsdict):
    if argsdict['dim']==1:
        #mode indices. axes appear in the order (l',l,m',m)
        mps=np.arange(-argsdict['Nx'],argsdict['Nx']+1)[np.newaxis,np.newaxis,:,np.newaxis]
        ms=np.arange(-argsdict['Nx'],argsdict['Nx']+1)[np.newaxis,np.newaxis,np.newaxis,:]

        kappax = argsdict['kx'] + argsdict['k1x']*ms
        kappa = np.abs(kappax)
        C = np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As'])
        S = -np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As'])

        F=kappa*(argsdict['g']+argsdict['sigma']/argsdict['rho']*kappa**2)*(1-((ms-mps)*(argsdict['k1x']*kappax))/kappa**2)*S
        G=-(1-((ms-mps)*(argsdict['k1x']*kappax))/kappa**2)*C

    elif argsdict['dim']==2:
        #mode indices. axes appear in the order (l',l,m',m,n',n)
        mps=np.arange(-argsdict['Nx'],argsdict['Nx']+1)[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
        ms=np.arange(-argsdict['Nx'],argsdict['Nx']+1)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
        nps=np.arange(-argsdict['Ny'],argsdict['Ny']+1)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
        ns=np.arange(-argsdict['Ny'],argsdict['Ny']+1)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]

        kappax = argsdict['kx'] + argsdict['k1x']*ms + argsdict['k2x']*ns
        kappay = argsdict['ky'] + argsdict['k1y']*ms + argsdict['k2y']*ns
        kappa = (kappax**2+kappay**2)**0.5

        C = np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']/2)*iv(ns-nps,kappa*argsdict['As']/2) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As']/2)*iv(ns-nps,-kappa*argsdict['As']/2)
        S = -np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']/2)*iv(ns-nps,kappa*argsdict['As']/2) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As']/2)*iv(ns-nps,-kappa*argsdict['As']/2)

        F=kappa*(argsdict['g']+argsdict['sigma']/argsdict['rho']*kappa**2)*(1-((ns-nps)*(argsdict['k1x']*kappax+argsdict['k1y']*kappay)+(ms-mps)*(argsdict['k2x']*kappax+argsdict['k2y']*kappay))/kappa**2)*S
        G=-(1-((ns-nps)*(argsdict['k1x']*kappax+argsdict['k1y']*kappay)+(ms-mps)*(argsdict['k2x']*kappax+argsdict['k2y']*kappay))/kappa**2)*C

    return F,G

def viscid_flat_boundary (omega, argsdict):
    if argsdict['dim']==1:
        # return array
        Ftilde = np.zeros((2, 2, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1),dtype=np.complex128)
        Gtilde = np.zeros((2, 2, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[:, np.newaxis]
        ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :]

        kappax = argsdict['kx']
        kappa = np.abs(kappax)
        Omega = 1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2
        C = np.exp(-kappa*argsdict['h0']) + np.exp(kappa*argsdict['h0'])

        Ctilde = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])  + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])

        S = - np.exp(-kappa*argsdict['h0'])  + np.exp(kappa*argsdict['h0'])

        Stilde = - np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])

        Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))

        Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
            (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
            4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[1, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,1:]

        Gtilde[1, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,:-1]

        Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)

        Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
            argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
            Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[1, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,1:]

        Gtilde[1, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,:-1]

        return Ftilde,Gtilde

    elif argsdict['dim']==2:
        # return array
        Ftilde = np.zeros((3, 3, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1),dtype=np.complex128)
        Gtilde = np.zeros((3, 3, 2*argsdict['Nt']+1, 2*argsdict['Nt']+1),dtype=np.complex128)

        # mode indices. axes appear in the order (l',l,m',m,n',n)
        lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[:, np.newaxis]
        ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :]

        kappax = argsdict['kx']
        kappay = argsdict['ky']
        kappa = (kappax**2+kappay**2)**0.5
        Omega = 1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2
        C = np.exp(-kappa*argsdict['h0']) + np.exp(kappa*argsdict['h0'])

        Ctilde = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])  + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])

        S = - np.exp(-kappa*argsdict['h0'])  + np.exp(kappa*argsdict['h0'])

        Stilde = - np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) + np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])

        Ftilde[0, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))

        Ftilde[1, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            2*argsdict['mu']*kappax*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)

        Ftilde[2, 0][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
            (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
            4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[2, 0][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,1:]

        Gtilde[2, 0][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,:-1]

        Ftilde[0, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            2*argsdict['mu']*kappax*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)

        Ftilde[1, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            (Ctilde - (2*argsdict['mu']*kappay**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))

        Ftilde[2, 1][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = -1j*kappay*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
            (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
            4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[2, 1][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = -1j*argsdict['rho']*kappay*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,1:]

        Gtilde[2, 1][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = -1j*argsdict['rho']*kappay*C/(4*(argsdict['rho']*Omega +
                                                                                                         argsdict['mu']*kappa**2))[:,:-1]

        Ftilde[0, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)

        Ftilde[1, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = 1j*np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0']) * \
            kappay*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)

        Ftilde[2, 2][(np.arange(Ftilde.shape[2]), np.arange(Ftilde.shape[3]))] = (-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
            argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
            Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

        Gtilde[2, 2][(np.arange(1, Ftilde.shape[2]), np.arange(Ftilde.shape[3]-1))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,1:]

        Gtilde[2, 2][(np.arange(Ftilde.shape[2]-1), np.arange(1, Ftilde.shape[3]))] = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + \
                                                                                                          argsdict['mu']*kappa**2))[:,:-1]

        return Ftilde,Gtilde

#Rayleigh quotient iteration for the viscid floquet problem
def rayleigh(omega_0, v0, w0, argsdict):
    vn=v0
    wn=w0

    omegas=[]
    vns=[]
    wns=[]
    omega=omega_0
    for n in range(argsdict['itmax']):
        E_n=viscid_mat(omega, argsdict)
        dE=(viscid_mat(omega+argsdict['domega_fd'],argsdict)-E_n)/argsdict['domega_fd']
        if argsdict['dim']==1:
            flat=np.transpose(E_n,(0,2,4,1,3,5)).reshape((2*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1),2*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)))
            bflat=np.einsum("kKlLmM,KLM",dE,vn).reshape(2*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1))
            btflat=np.einsum("kKlLmM,klm",dE,wn).reshape(2*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1))
        if argsdict['dim']==2:
            flat=np.transpose(E_n,(0,2,4,6,1,3,5,7)).reshape((3*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1),3*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1)))
            bflat=np.einsum("kKlLmMnN,KLMN",dE,vn).reshape(3*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1))
            btflat=np.einsum("kKlLmMnN,klmn",dE,wn).reshape(3*(2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1))
        try:
            xi=solve(flat, bflat)
            zeta=solve(flat.T, btflat)
            if argsdict['dim']==1:
                xi=xi.reshape(2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1))
                zeta=zeta.reshape(2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1))
                domega=-np.einsum("kKlLmM,KLM,klm",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn)
            if argsdict['dim']==2:
                xi=xi.reshape(3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1))
                zeta=zeta.reshape(3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1))
                domega=-np.einsum("kKlLmMnN,KLMN,klmn",E_n,vn,wn)/np.einsum("kKlLmMnN,KLMN,klmn",dE,vn,wn)

            omega=omega+domega
            vn=(xi/np.linalg.norm(xi))
            wn=zeta/np.linalg.norm(zeta)
            omegas=omegas+[omega]
            vns=vns+[vn]
            wns=wns+[wn]
        # except LinAlgWarning:
        except LinAlgError:
            if argsdict['dim']==1:
                domega=-np.einsum("kKlLmM,KLM,klm",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn)
            if argsdict['dim']==2:
                domega=-np.einsum("kKlLmMnN,KLMN,klmn",E_n,vn,wn)/np.einsum("kKlLmMnN,KLMN,klmn",dE,vn,wn)

            omega=omega+domega
            omegas=omegas+[omega]
            vns=vns+[vn]
            wns=wns+[wn]
            break

    if argsdict['dim']==1:
        print(n, omega, np.einsum("kKlLmM,KLM,klm",E_n,vn,wn)/np.einsum("kKlLmM,KLM,klm",dE,vn,wn))
    if argsdict['dim']==2:
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
    argsdict=args.__dict__

    if not argsdict['dim']==1 and not argsdict['dim']==2:
        print('Dimension must be 1 or 2')
        exit(0)

    #If no initial modes exist, use the lowest frequency inviscid modes to start
    print(argsdict)
    json=json.dumps(argsdict)
    f=open(argsdict['filebase']+'argsdict.json','w')
    f.write(json)
    f.close()
    As=argsdict['As']
    argsdict['As']=0
    F,G=inviscid_mat(argsdict)
    if argsdict['dim']==1:
        Fflattened=F[0,0]
        Gflattened=G[0,0]
    elif argsdict['dim']==2:
        Fflattened=np.transpose(F[0,0],(0,2,1,3)).reshape(((2*argsdict['Nx']+1)*(2*argsdict['Ny']+1),(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1)))
        Gflattened=np.transpose(G[0,0],(0,2,1,3)).reshape(((2*argsdict['Nx']+1)*(2*argsdict['Ny']+1),(2*argsdict['Nx']+1)*(2*argsdict['Ny']+1)))
    evals,levecs,revecs=eig(Fflattened,Gflattened,left=True)

    inds=np.argsort(-evals)[:argsdict['nmodes']]
    omegas=[]
    vns=[]
    wns=[]
    for i in range(argsdict['nmodes']):
        ind=inds[i]
        if argsdict['dim']==1:
            v0_inviscid=revecs[:,ind]
            # w0_inviscid=np.conjugate(levecs[:,ind])
            w0_inviscid=1
            v=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)
            w=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)

        elif argsdict['dim']==2:
            v0_inviscid=revecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1)))
            # w0_inviscid=np.conjugate(levecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1))))
            w0_inviscid=1
            v=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
            w=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)

        omega_inviscid=(-evals[ind])**0.5
        print(omega_inviscid)
        if argsdict['dim']==1:
            v[1,argsdict['Nt']]=v0_inviscid
            w[1,argsdict['Nt']]=w0_inviscid
        if argsdict['dim']==2:
            v[2,argsdict['Nt']]=v0_inviscid
            w[2,argsdict['Nt']]=w0_inviscid


        #Should we iteratively increase as through num steps here??
        # omegas_i,vns_i,wns_i=rayleigh(omega_inviscid,v,w,argsdict)
        omega=omega_inviscid
        argsdict['As']=0
        for it in range(argsdict['num']):
            argsdict['As'] += As/argsdict['num']
            print(argsdict['As'])
            omegas_i,vns_i,wns_i=rayleigh(omega,v,w,argsdict)
            omega=omegas_i[-1]
            v=vns_i[-1]
            w=wns_i[-1]
        print(argsdict['As'])
        omegas=omegas+[omegas_i[-1]]
        vns=vns+[vns_i[-1]]
        wns=wns+[wns_i[-1]]
    if argsdict['negatives']:
        for i in range(argsdict['nmodes']):
            ind=inds[i]
            if argsdict['dim']==1:
                v0_inviscid=revecs[:,ind]
                # w0_inviscid=np.conjugate(levecs[:,ind])
                w0_inviscid=1
                v=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)
                w=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)

            elif argsdict['dim']==2:
                v0_inviscid=revecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1)))
                # w0_inviscid=np.conjugate(levecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1))))
                w0_inviscid=1
                v=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
                w=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
            omega_inviscid=-(-evals[ind])**0.5
            print(omega_inviscid)
            if argsdict['dim']==1:
                v[1,argsdict['Nt']]=v0_inviscid
                w[1,argsdict['Nt']]=w0_inviscid
            if argsdict['dim']==2:
                v[2,argsdict['Nt']]=v0_inviscid
                w[2,argsdict['Nt']]=w0_inviscid

            argsdict['As']=0
            omega=omega_inviscid
            for it in range(argsdict['num']):
                argsdict['As'] += As/argsdict['num']
                print(argsdict['As'])
                omegas_i,vns_i,wns_i=rayleigh(omega,v,w,argsdict)
                omega=omegas_i[-1]
                v=vns_i[-1]
                w=wns_i[-1]
            omegas=omegas+[omegas_i[-1]]
            vns=vns+[vns_i[-1]]
            wns=wns+[wns_i[-1]]
    #save the matrices to with the filebase prefix
    np.save(argsdict['filebase']+"evals.npy",np.array(omegas))
    np.save(argsdict['filebase']+"evecs.npy",[np.array(vns),np.array(wns)])

    #Continuation along ad.
    ad0=argsdict['ad']
    omegas=np.load(argsdict['filebase']+'evals.npy')
    revecs,levecs=np.load(argsdict['filebase']+'evecs.npy')
    for i in range(len(omegas)):
        argsdict['ad']=ad0
        omegan=[omegas[i]]
        vn=[revecs[i]]
        wn=[levecs[i]]
        omegans=[omegan[-1]]
        vns=[vn[-1]]
        wns=[wn[-1]]

        for it in range(argsdict['num']):
            argsdict['ad'] += argsdict['g']*argsdict['dad']/argsdict['num']
            print(argsdict['ad'])
            omegan,vn,wn=rayleigh(omegan[-1],vn[-1],wn[-1],argsdict)
            omegans=omegans+[omegan[-1]]
            vns=vns+[vn[-1]]
            wns=wns+[wn[-1]]

        np.save(argsdict['filebase']+"evals_cont"+str(i)+".npy",np.array(omegans))
        np.save(argsdict['filebase']+"evecs_cont"+str(i)+".npy",[np.array(vns),np.array(wns)])

else:
    #For benchmarking in jupyter
    class argsdict:
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
