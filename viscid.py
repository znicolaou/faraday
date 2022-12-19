#!/usr/bin/env python
import numpy as np
import argparse
import json
import timeit
from scipy.linalg import eig
from scipy.special import iv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye


def makeindices (argsdict):
    if argsdict['dim']==1:
        #l lp m mp k kp
        ones=np.ones((2*argsdict['Nt']+1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Nx']+1,2,2,1),dtype=int)
        indices=np.concatenate([ones*np.arange(2*argsdict['Nt']+1).reshape((2*argsdict['Nt']+1,1,1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Nt']+1).reshape((1,2*argsdict['Nt']+1,1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Nx']+1).reshape((1,1,2*argsdict['Nx']+1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Nx']+1).reshape((1,1,1,2*argsdict['Nx']+1,1,1,1)),
                                ones*np.arange(2).reshape((1,1,1,1,2,1,1)),
                                ones*np.arange(2).reshape((1,1,1,1,1,2,1))],axis=-1)
        diag=(np.arange(2*argsdict['Nt']+1),np.arange(2*argsdict['Nt']+1))
        diag_inds=indices[diag]
        #k l m kp lp mp
        d=[diag_inds[...,i,j,:].transpose((1,0,2,3)).reshape(((2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)**2,6))[:,(4,0,2,5,1,3)] for i in range(2) for j in range(2)]
        diag_rows=np.concatenate([[np.ravel_multi_index(d,(2,2*argsdict['Nt']+1,2*argsdict['Nx']+1)) for d in d[i][:,:3]] for i in range(len(d))])
        diag_cols=np.concatenate([[np.ravel_multi_index(d,(2,2*argsdict['Nt']+1,2*argsdict['Nx']+1)) for d in d[i][:,3:]] for i in range(len(d))])

        udiag=(np.arange(1,2*argsdict['Nt']+1),np.arange(2*argsdict['Nt']))
        diag_inds=indices[udiag]
        d=[diag_inds[...,i,j,:].transpose((1,0,2,3)).reshape(((2*argsdict['Nt'])*(2*argsdict['Nx']+1)**2,6))[:,(4,0,2,5,1,3)] for i in range(1,2) for j in range(2)]
        udiag_rows=np.concatenate([[np.ravel_multi_index(d,(2,2*argsdict['Nt']+1,2*argsdict['Nx']+1)) for d in d[i][:,:3]] for i in range(len(d))])
        udiag_cols=np.concatenate([[np.ravel_multi_index(d,(2,2*argsdict['Nt']+1,2*argsdict['Nx']+1)) for d in d[i][:,3:]] for i in range(len(d))])


        ldiag=(np.arange(2*argsdict['Nt']),np.arange(1,2*argsdict['Nt']+1))
        diag_inds=indices[ldiag]
        d=[diag_inds[...,i,j,:].transpose((1,0,2,3)).reshape(((2*argsdict['Nt'])*(2*argsdict['Nx']+1)**2,6))[:,(4,0,2,5,1,3)] for i in range(1,2) for j in range(2)]
        ldiag_rows=np.concatenate([[np.ravel_multi_index(d,(2,2*argsdict['Nt']+1,2*argsdict['Nx']+1)) for d in d[i][:,:3]] for i in range(len(d))])
        ldiag_cols=np.concatenate([[np.ravel_multi_index(d,(2,2*argsdict['Nt']+1,2*argsdict['Nx']+1)) for d in d[i][:,3:]] for i in range(len(d))])

        #k l m kp lp mp
        lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        mps = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        ms = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        indices=[lps,ls,mps,ms]
    elif argsdict['dim']==2:
        #l lp m mp n, np, k kp
        ones=np.ones((2*argsdict['Nt']+1,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Nx']+1, 2*argsdict['Ny']+1, 2*argsdict['Ny']+1, 3, 3, 1),dtype=int)
        indices=np.concatenate([ones*np.arange(2*argsdict['Nt']+1).reshape((2*argsdict['Nt']+1,1,1,1,1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Nt']+1).reshape((1,2*argsdict['Nt']+1,1,1,1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Nx']+1).reshape((1,1,2*argsdict['Nx']+1,1,1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Nx']+1).reshape((1,1,1,2*argsdict['Nx']+1,1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Ny']+1).reshape((1,1,1,1,2*argsdict['Nx']+1,1,1,1,1)),
                                ones*np.arange(2*argsdict['Ny']+1).reshape((1,1,1,1,1,2*argsdict['Nx']+1,1,1,1)),
                                ones*np.arange(3).reshape((1,1,1,1,1,1,3,1,1)),
                                ones*np.arange(3).reshape((1,1,1,1,1,1,1,3,1))],axis=-1)
        diag=(np.arange(2*argsdict['Nt']+1),np.arange(2*argsdict['Nt']+1))
        diag_inds=indices[diag]

        #k l m n kp lp mp np
        d=[diag_inds[...,i,j,:].transpose((1,3,0,2,4,5)).reshape(((2*argsdict['Nt']+1)*(2*argsdict['Nx']+1)**2*(2*argsdict['Ny']+1)**2,8))[:,(6,0,2,4,7,1,3,5)] for i in range(3) for j in range(3)]
        diag_rows=np.concatenate([[np.ravel_multi_index(d,(3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1)) for d in d[i][:,:4]] for i in range(len(d))])
        diag_cols=np.concatenate([[np.ravel_multi_index(d,(3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1)) for d in d[i][:,4:]] for i in range(len(d))])

        udiag=(np.arange(1,2*argsdict['Nt']+1),np.arange(2*argsdict['Nt']))
        diag_inds=indices[udiag]
        d=[diag_inds[...,i,j,:].transpose((1,3,0,2,4,5)).reshape(((2*argsdict['Nt'])*(2*argsdict['Nx']+1)**2*(2*argsdict['Ny']+1)**2,8))[:,(6,0,2,4,7,1,3,5)] for i in range(2,3) for j in range(3)]
        udiag_rows=np.concatenate([[np.ravel_multi_index(d,(3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1)) for d in d[i][:,:4]] for i in range(len(d))])
        udiag_cols=np.concatenate([[np.ravel_multi_index(d,(3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1)) for d in d[i][:,4:]] for i in range(len(d))])

        ldiag=(np.arange(2*argsdict['Nt']),np.arange(1,2*argsdict['Nt']+1))
        diag_inds=indices[ldiag]
        d=[diag_inds[...,i,j,:].transpose((1,3,0,2,4,5)).reshape(((2*argsdict['Nt'])*(2*argsdict['Nx']+1)**2*(2*argsdict['Ny']+1)**2,8))[:,(6,0,2,4,7,1,3,5)] for i in range(2,3) for j in range(3)]
        ldiag_rows=np.concatenate([[np.ravel_multi_index(d,(3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1)) for d in d[i][:,:4]] for i in range(len(d))])
        ldiag_cols=np.concatenate([[np.ravel_multi_index(d,(3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1)) for d in d[i][:,4:]] for i in range(len(d))])

        lps = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ls = np.arange(-argsdict['Nt'], argsdict['Nt'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        mps = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ms = np.arange(-argsdict['Nx'], argsdict['Nx'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        nps = np.arange(-argsdict['Ny'], argsdict['Ny'] + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ns = np.arange(-argsdict['Ny'], argsdict['Ny'] + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        indices=[lps,ls,mps,ms,nps,ns]

    return indices, np.concatenate([diag_rows, udiag_rows, ldiag_rows]), np.concatenate([diag_cols, udiag_cols, ldiag_cols])

#Return a numpy array corresponding to E^{klmn}_{k'l'm'n'} in Eqs. 31-39 in the notes. Return array has shape (3,3,2*Nt+1,2*Nt+1,2*Nx+1,2*Nx+1,2*Ny+1,2*Ny+1), with axes correponding to (k',k,l',l,m',m,n',n).
def viscid_mat1d (omega, argsdict):
    lps,ls,mps,ms=indices
    kappax = argsdict['kx'] + argsdict['k1x']*ms
    kappa = np.abs(kappax)
    Omega = 1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2
    EpOmega=np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])
    EmOmega=np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])
    C = np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As'])
    Ctilde = EmOmega * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']) + EpOmega * iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As'])
    S = -np.exp(-kappa*argsdict['h0'])*iv(ms-mps,kappa*argsdict['As']) + np.exp(kappa*argsdict['h0'])*iv(ms-mps,-kappa*argsdict['As'])
    Stilde = - EmOmega * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']) + EpOmega * iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As'])

    vals00=EmOmega * \
                (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))
    vals01=1j*EmOmega * \
                kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)
    vals10=-1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
                (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
                4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])
    vals11=(-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
                argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
                Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

    uvals10 = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,1:]
    lvals10 = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:-1]
    uvals11 = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,1:]
    lvals11 = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:-1]

    n_flat=np.product((2,2*argsdict['Nt']+1,2*argsdict['Nx']+1))
    vals=np.concatenate([vals00.ravel(),vals01.ravel(),vals10.ravel(),vals11.ravel(),-argsdict['g']*argsdict['ad']*uvals10.ravel(),-argsdict['g']*argsdict['ad']*uvals11.ravel(),-argsdict['g']*argsdict['ad']*lvals10.ravel(),-argsdict['g']*argsdict['ad']*lvals11.ravel()])
    E=csr_matrix((vals,(rows,cols)),shape=(n_flat,n_flat))

    return E

def viscid_flatundriven_mat1d (omega, argsdict):
    kappax = argsdict['kx']
    kappa = np.abs(kappax)
    Omega = 1j*(omega) + argsdict['mu']/argsdict['rho']*kappa**2
    EpOmega=np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])
    EmOmega=np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])
    C = np.exp(-kappa*argsdict['h0']) + np.exp(kappa*argsdict['h0'])
    Ctilde = EmOmega + EpOmega
    S = -np.exp(-kappa*argsdict['h0']) + np.exp(kappa*argsdict['h0'])
    Stilde = - EmOmega  + EpOmega

    vals00=EmOmega * \
                (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))
    vals01=1j*EmOmega * \
                kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)
    vals10=-1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
                (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
                4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])
    vals11=(-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
                argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
                Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

    E=np.array([[vals00,vals01],[vals10,vals11]])

    return csr_matrix(E)

def viscid_mat2d (omega, argsdict):
    lps,ls,mps,ms,nps,ns=indices
    kappax = argsdict['kx'] + argsdict['k1x']*ms + argsdict['k2x']*ns
    kappay = argsdict['ky'] + argsdict['k1y']*ms + argsdict['k2y']*ns
    kappa = (kappax**2+kappay**2)**0.5
    C = np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + \
        np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5)
    S = - np.exp(-kappa*argsdict['h0']) * iv(ms-mps, kappa*argsdict['As']*0.5) * iv(ns-nps, kappa*argsdict['As']*0.5) + \
        np.exp(kappa*argsdict['h0']) * iv(ms-mps, -kappa*argsdict['As']*0.5) * iv(ns-nps, -kappa*argsdict['As']*0.5)

    Omega = 1j*(omega + 2*np.pi*argsdict['freq']*ls) + argsdict['mu']/argsdict['rho']*kappa**2
    EpOmega=np.exp((argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])
    EmOmega=np.exp(-(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['h0'])

    Ctilde = EmOmega * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * \
             iv(ns-nps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) + EpOmega * \
             iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5)
    Stilde = - EmOmega * iv(ms-mps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * \
             iv(ns-nps, (argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) + EpOmega * \
             iv(ms-mps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5) * iv(ns-nps, -(argsdict['rho']/argsdict['mu']*Omega)**0.5*argsdict['As']*0.5)

    vals00 = EmOmega * \
        (Ctilde - (2*argsdict['mu']*kappax**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))
    vals10 = -EmOmega * \
        2*argsdict['mu']*kappax*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)
    vals20 = -1j*kappax*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
        (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
        4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])
    vals01 = -EmOmega * \
        2*argsdict['mu']*kappax*kappay*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)
    vals11 = EmOmega * \
        (Ctilde - (2*argsdict['mu']*kappay**2*C)/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))
    vals21 = -1j*kappay*(2*argsdict['mu']*(argsdict['rho']/argsdict['mu']*Omega)**0.5 * \
        (Ctilde-Stilde) + (argsdict['rho']*Omega+argsdict['mu']*kappa**2)*S/kappa + (argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma'] - \
        4*argsdict['mu']**2*(argsdict['rho']*Omega/argsdict['mu'])**0.5))*C/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])
    vals02 = 1j*EmOmega * \
        kappax*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)
    vals12 = 1j*EmOmega * \
        kappay*(-2*argsdict['mu']*kappa*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2)+Stilde/(argsdict['rho']*Omega/argsdict['mu'])**0.5)
    vals22 = (-2*argsdict['mu']*kappa**2*(Ctilde-Stilde)+(argsdict['rho']*Omega + \
        argsdict['mu']*kappa**2)*C+kappa*(argsdict['g']*argsdict['rho']**2+kappa**2*(argsdict['rho']*argsdict['sigma']-4*argsdict['mu']**2*(argsdict['rho'] * \
        Omega/argsdict['mu'])**0.5))*S/(argsdict['rho']*Omega+argsdict['mu']*kappa**2))/(2*argsdict['rho'])

    uvals20 = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:,1:]
    lvals20 = -1j*argsdict['rho']*kappax*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:,:-1]
    uvals21 = -1j*argsdict['rho']*kappay*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:,1:]
    lvals21 = -1j*argsdict['rho']*kappay*C/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:,:-1]
    uvals22 = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:,1:]
    lvals22 = argsdict['rho']*kappa*S/(4*(argsdict['rho']*Omega + argsdict['mu']*kappa**2))[:,:,:,:,:,:-1]

    n_flat=np.product((3,2*argsdict['Nt']+1,2*argsdict['Nx']+1,2*argsdict['Ny']+1))
    vals=np.concatenate([vals00.ravel(),vals01.ravel(),vals02.ravel(),vals10.ravel(),vals11.ravel(),vals12.ravel(),vals20.ravel(),vals21.ravel(),vals22.ravel(),-argsdict['g']*argsdict['ad']*uvals20.ravel(),-argsdict['g']*argsdict['ad']*uvals21.ravel(),-argsdict['g']*argsdict['ad']*uvals22.ravel(),-argsdict['g']*argsdict['ad']*lvals20.ravel(),-argsdict['g']*argsdict['ad']*lvals21.ravel(),-argsdict['g']*argsdict['ad']*lvals22.ravel()])
    E=csr_matrix((vals,(rows,cols)),shape=(n_flat,n_flat))

    return E

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

#Rayleigh quotient iteration
def rayleigh_mat(omega_0, v0, w0, mat, argsdict):
    vn=v0
    wn=w0

    omegas=[omega_0]
    vns=[v0]
    wns=[w0]
    omega=omega_0
    for n in range(argsdict['itmax']):
        E_n=mat(omega, argsdict)
        dE=(mat(omega+argsdict['domega_fd'],argsdict)-E_n)/argsdict['domega_fd']
        xi=spsolve(E_n, dE.dot(vn))
        zeta=spsolve(E_n.T, dE.T.dot(wn))
        domega=-E_n.dot(vn).dot(wn)/dE.dot(vn).dot(wn)

        omega=omega+domega
        vn=xi/np.linalg.norm(xi)
        wn=zeta/np.linalg.norm(zeta)
        omegas=omegas+[omega]
        vns=vns+[vn]
        wns=wns+[wn]

        E_n=mat(omega, argsdict)
        f1=np.linalg.norm(E_n.dot(vn))
        f2=np.linalg.norm(E_n.T.dot(wn))
        dv=np.abs(vns[n+1])-np.abs(vns[n])
        dw=np.abs(wns[n+1])-np.abs(wns[n])
        verr=np.linalg.norm(dv,ord=np.inf)/(1+np.linalg.norm(vn,ord=np.inf))
        werr=np.linalg.norm(dw,ord=np.inf)/(1+np.linalg.norm(wn,ord=np.inf))
        omegaerr=np.abs(domega)/(1+np.abs(omega))
        if argsdict['verbose']>0:
            print("n=%i dv=%e dw=%e dl=%e f1=%e f2=%e"%(n, verr, werr, omegaerr, f1, f2))
        if verr<argsdict['epsu'] and werr<argsdict['epsu'] and omegaerr<argsdict['epsl'] and f1<argsdict['epsf'] and f2<argsdict['epsf']:
            break

    return omegas,vns,wns


def cont(omegas, vs, ws, mat, argsdict):
    argsdict[argsdict[argsdict['par']]]=argsdict['pari']

    omegans=[]
    vns=[]
    wns=[]
    parns=[]
    for i in range(len(omegas)):
        omegan=[omegas[i]]
        vn=[vs[i]]
        wn=[ws[i]]
        omegans=omegans+[[omegan[-1]]]
        vns=vns+[[vn[-1]]]
        wns=wns+[[wn[-1]]]
    parns=[argsdict[argsdict['par']]]

    ds = argsdict['ds']

    scount=0
    fail=False
    s=0
    try:
        while s<1:
            if s+ds>1:
                ds=1-s
            s += ds
            dmu = (argsdict['parf']-argsdict['pari'])*ds

            argsdict[argsdict['par']] += dmu

            its=0
            parns=parns+[argsdict[argsdict['par']]]
            for i in range(len(omegas)):
                #we could try to add a predictor based on jacobians here
                #deltaomega = -(w E_mu v/w E_omega v)*dmu
                #E deltav = -(E_mu deltamu + E_omega deltaomega)v
                #E.T detlaw = -(E.T_mu deltamu +E.T_omga deltaomega)w
                E=mat(omegans[i][-1],argsdict)
                E_omega = (mat(omegans[i][-1]+argsdict['domega_fd'],argsdict)-E)/argsdict['domega_fd']
                argsdict[argsdict['par']] += argsdict['dmu_fd']
                E_mu = (mat(omegans[i][-1],argsdict)-E)/argsdict['dmu_fd']
                argsdict[argsdict['par']] -= argsdict['dmu_fd']
                domega = -(wns[i][-1].dot(E_mu.dot(vns[i][-1])))/(wns[i][-1].dot(E_omega.dot(vns[i][-1]))) * dmu
                dv=-spsolve(E, -(E_mu*dmu + E_omega*domega).dot(vns[i][-1]))
                dw=-spsolve(E.T, -(E_mu.T*dmu + E_omega.T*domega).dot(wns[i][-1]))

                # omegan,vn,wn=rayleigh_mat(omegans[i][-1],vns[i][-1],wns[i][-1],mat,argsdict)
                omegan,vn,wn=rayleigh_mat(omegans[i][-1]+domega,vns[i][-1]+dv,wns[i][-1]+dw,mat,argsdict)

                omegans[i]=omegans[i]+[omegan[-1]]
                vns[i]=vns[i]+[vn[-1]]
                wns[i]=wns[i]+[wn[-1]]
                if(len(omegan)>its):
                    its=len(omegan)
                if len(omegan)>argsdict['itmax']:
                    fail=True
            if not fail:
                for i in range(len(omegas)-1):
                    for j in range(i+1,len(omegas)):
                        diffr=np.mod(np.real(omegans[i][-1]-omegans[j][-1])+np.pi*argsdict['freq'],2*np.pi*argsdict['freq'])-np.pi*argsdict['freq']
                        diffi=np.imag(omegans[i][-1]-omegans[j][-1])
                        diff=np.abs(diffr+1j*diffi)

                        if np.abs(omegans[i][-1]-omegans[j][-1]) < argsdict['epss']:
                            print('Subharmonic degenerate! '+str(i))
                            omega=omegans[i][-2]
                            # v=vns[i][-2]
                            # w=wns[i][-2]
                            def mat_deflate(omega,argsdict):
                                E=mat(omega,argsdict)
                                x=vns[j][-1]
                                y=wns[j][-1]/np.conjugate(vns[j][-1].dot(wns[j][-1].conjugate()))
                                prod=csr_matrix(y[:,np.newaxis]*x[np.newaxis,:].conjugate())
                                return E.dot(eye(*E.shape)-(omega-omegans[j][-1]-1)/(omega-omegans[j][-1])*prod)

                            v=vns[i][-2] - wns[j][-1].conjugate()*np.sum(vns[i][-2]*wns[j][-1].conjugate())/np.sum(wns[j][-2]*wns[j][-1].conjugate())
                            w=wns[i][-2] - vns[j][-1].conjugate()*np.sum(wns[i][-2]*vns[j][-1].conjugate())/np.sum(vns[j][-2]*vns[j][-1].conjugate())
                            # argsdict['verbose']=1
                            # omegan,vn,wn=rayleigh_mat(omega,v,w,mat_deflate,argsdict)
                            omegan,vn,wn=rayleigh_mat(omega,v,w,mat,argsdict)
                            # argsdict['verbose']=0
                            omegans[i][-1]=omegan[-1]
                            vns[i][-1]=vn[-1]
                            wns[i][-1]=wn[-1]
                            print(omegans[i][-1],omegans[j][-1])

                        elif diff < argsdict['epss']:
                            print('Harmonic degenerate! '+str(i))
                            # omega=omegans[i][-2]
                            # v=vns[i][-2] - wns[j][-1].conjugate()*np.sum(vns[i][-2]*wns[j][-1].conjugate())/np.sum(wns[j][-2]*wns[j][-1].conjugate())
                            # w=wns[i][-2] - vns[j][-1].conjugate()*np.sum(wns[i][-2]*vns[j][-1].conjugate())/np.sum(vns[j][-2]*vns[j][-1].conjugate())
                            # omegan,vn,wn=rayleigh_mat(omega,v,w,mat,argsdict)
                            # omegans[i][-1]=omegan[-1]
                            # vns[i][-1]=vn[-1]
                            # wns[i][-1]=wn[-1]

                        if np.abs(omegans[i][-1]-omegans[j][-1]) < argsdict['epss']:
                            fail=True #did not find new branch

                for i in range(len(omegas)):
                    dv=(np.abs(vns[i][-1])-np.abs(vns[i][-2])).reshape(np.product(vns[i][-1].shape))
                    dw=(np.abs(wns[i][-1])-np.abs(wns[i][-2])).reshape(np.product(wns[i][-1].shape))
                    domega=omegans[i][-1]-omegans[i][-2]

                    verr=np.linalg.norm(dv,ord=np.inf)/(1+np.linalg.norm(vns[i][-1].reshape(np.product(vns[i][-1].shape)),ord=np.inf))
                    werr=np.linalg.norm(dw,ord=np.inf)/(1+np.linalg.norm(wns[i][-1].reshape(np.product(vns[i][-1].shape)),ord=np.inf))

                    omegaerr=np.abs(domega)/(1+np.abs(omegan[-1]))

                    if argsdict['verbose']>0:
                        print("dv=",verr,"dw=",werr,"domega=",omegaerr)
                    if verr>argsdict['dumax'] or werr>argsdict['dumax'] or omegaerr>argsdict['dlmax']:
                        fail=True

            if fail:
                if argsdict['verbose']>0:
                    print('Convergence failure!')
                fail=False
                scount=0
                s -= ds
                argsdict[argsdict['par']] -= (argsdict['parf']-argsdict['pari'])*ds
                ds=ds/2
                for i in range(len(omegas)):
                    omegans[i]=omegans[i][:-1]
                    vns[i]=vns[i][:-1]
                    wns[i]=wns[i][:-1]
                parns=parns[:-1]
                if ds<argsdict['dsmin']:
                    if argsdict['verbose']>0:
                        print('Minimum step!')
                    imin=np.min([len(omegans[i]) for i in range(len(omegas))])
                    parns=parns[:imin]
                    for i in range(len(omegas)):
                        omegans[i]=omegans[i][:imin]
                        vns[i]=vns[i][:imin]
                        wns[i]=wns[i][:imin]
                    return omegans,vns,wns,parns
            else:
                print(argsdict[argsdict['par']], s, ds, its)
                scount=scount+1
                if scount>4 and ds*2<argsdict['dsmax']:
                    ds=ds*2
                    scount=0
    except KeyboardInterrupt:
        print("Keyboard interrupt!")
        imin=np.min([len(omegans[i]) for i in range(len(omegas))])
        parns=parns[:imin]
        for i in range(len(omegas)):
            omegans[i]=omegans[i][:imin]
            vns[i]=vns[i][:imin]
            wns[i]=wns[i][:imin]
    return omegans,vns,wns,parns


#include this in case we want to import functions here elsewhere, in a jupyter notebook for example.
if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser(description='Find the Floquet exponents for a given drive and wavenumber.')
    parser.add_argument("--filebase", type=str, required=True, dest='filebase', help='Base string for file output.')
    parser.add_argument("--frequency", type=float, required=False, default=5, dest='freq', help='Driving frequency (Hz)')
    parser.add_argument("--ad", type=float, required=False, default=0.0, dest='ad', help='Driving acceleration (in gravitational units)')
    parser.add_argument("--viscosity", type=float, required=False, default=0.005, dest='mu', help='Viscosity (cgs units)')
    parser.add_argument("--density", type=float, required=False, default=1.0, dest='rho', help='Fluid density (cgs units)')
    parser.add_argument("--gravity", type=float, required=False, default=980, dest='g', help='Gravitational acceleration (cgs units)')
    parser.add_argument("--tension", type=float, required=False, default=20, dest='sigma', help='Surface tension (cgs units)')
    parser.add_argument("--kx", type=float, required=False, default=0.45*np.pi, dest='kx', help='Wave vector x component')
    parser.add_argument("--ky", type=float, required=False, default=0, dest='ky', help='Wave vector y component')
    parser.add_argument("--height", type=float, required=False, default=0.1, dest='h0', help='Fluid depth')
    parser.add_argument("--As", type=float, required=False, default=0.0, dest='As', help='Substrate height')
    parser.add_argument("--k1x", type=float, required=False, default=np.pi, dest='k1x', help='Second reciprocal lattice vector x component')
    parser.add_argument("--k1y", type=float, required=False, default=0, dest='k1y', help='Second reciprocal lattice vector y component')
    parser.add_argument("--k2x", type=float, required=False, default=-0.5*np.pi, dest='k2x', help='First reciprocal lattice vector x component')
    parser.add_argument("--k2y", type=float, required=False, default=3**0.5/2*np.pi, dest='k2y', help='First reciprocal lattice vector y component')
    parser.add_argument("--Nt", type=int, required=False, default=5, dest='Nt', help='Number of modes to include the Floquet expansion for time')
    parser.add_argument("--Nx", type=int, required=False, default=5, dest='Nx', help='Number of modes to include the Floquet expansion for spatial x')
    parser.add_argument("--Ny", type=int, required=False, default=5, dest='Ny', help='Number of modes to include the Floquet expansion for spatial y')
    parser.add_argument("--dim", type=int, required=False, default=2, dest='dim', help='Dimension, 1 or 2. Default 2.')
    parser.add_argument("--itmax", type=int, required=False, default=100, dest='itmax', help='Number of iterators in acceleration continuation.')
    parser.add_argument("--ds", type=float, required=False, default=1e-3, dest='ds', help='Initial parameter change')
    parser.add_argument("--dsmax", type=float, required=False, default=1e-1, dest='dsmax', help='Max parameter change')
    parser.add_argument("--dsmin", type=float, required=False, default=1e-10, dest='dsmin', help='Min parameter change')
    parser.add_argument("--domega_fd", type=float, required=False, default=1e-3, dest='domega_fd', help='Finite difference step')
    parser.add_argument("--dmu_fd", type=float, required=False, default=1e-3, dest='dmu_fd', help='Finite difference step')
    parser.add_argument("--epsu", type=float, required=False, default=1e-4, dest='epsu', help='Newton tolerance for state variables')
    parser.add_argument("--epsf", type=float, required=False, default=1e-1, dest='epsf', help='Newton tolerance for state variables')
    parser.add_argument("--epsl", type=float, required=False, default=1e-6, dest='epsl', help='Newton tolerance for parameters')
    parser.add_argument("--epss", type=float, required=False, default=1e-4, dest='epss', help='Tolerance for degeneracies')
    parser.add_argument("--dumax", type=float, required=False, default=1e-1, dest='dumax', help='Max state change')
    parser.add_argument("--dlmax", type=float, required=False, default=1e-2, dest='dlmax', help='Max parameter change')
    parser.add_argument("--nmodes", type=int, required=False, default=1, dest='nmodes', help='Number of modes to track. Default 2.')
    parser.add_argument("--negatives", type=int, required=False, default=1, dest='negatives', help='Include negative frequencies')
    parser.add_argument("--verbose", type=int, required=False, default=False, dest='verbose', help='Print interation details')
    parser.add_argument("--sweeps", type=str, required=False, default='', dest='sweeps', help='List of parameter sweeps')
    parser.add_argument("--continue", type=int, required=False, default=0, dest='cont', help='Flag to continue. If zero, generate initial conditions from inviscid modes. If one, use the last entries in filebaseevals.npy and filebaseevecs.npy. Default 0.')

    args = parser.parse_args()
    argsdict=args.__dict__

    if not argsdict['dim']==1 and not argsdict['dim']==2:
        print('Dimension must be 1 or 2')
        exit(0)

    indices, rows, cols=makeindices(argsdict)
    start=timeit.default_timer()
    print(argsdict)
    if argsdict['cont']==0:
        #If no initial modes exist, use the lowest frequency inviscid modes to start
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
        vs=[]
        ws=[]
        for i in range(argsdict['nmodes']):
            ind=inds[i]
            #put the mode in the first brillouin zone
            omega_inviscid=np.mod(np.real((-evals[ind])**0.5),2*np.pi*argsdict['freq'])
            offset=np.floor(np.real((-evals[ind])**0.5)/(2*np.pi*argsdict['freq'])).astype(int)
            if argsdict['dim']==1:
                v0_inviscid=revecs[:,ind]
                w0_inviscid=np.conjugate(levecs[:,ind])
                v=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)
                w=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)
                v[1,argsdict['Nt']+offset]=v0_inviscid
                w[1,argsdict['Nt']+offset]=w0_inviscid
                mat=viscid_mat1d

            elif argsdict['dim']==2:
                v0_inviscid=revecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1)))
                w0_inviscid=np.conjugate(levecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1))))
                v=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
                w=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
                v[2,argsdict['Nt']+offset]=v0_inviscid
                w[2,argsdict['Nt']+offset]=w0_inviscid
                mat=viscid_mat2d

            omega=omega_inviscid
            argsdict['As']=0
            omegas_i,vs_i,ws_i=rayleigh_mat(omega,v.ravel(),w.ravel(),mat,argsdict)
            print(omega_inviscid, omegas_i[-1])
            omegas=omegas+[omegas_i[-1]]
            vs=vs+[vs_i[-1]]
            ws=ws+[ws_i[-1]]

        if argsdict['negatives']:
            for i in range(argsdict['nmodes']):
                ind=inds[i]
                omega_inviscid=np.mod(-np.real((-evals[ind])**0.5),2*np.pi*argsdict['freq'])
                offset=np.floor(-np.real((-evals[ind])**0.5)/(2*np.pi*argsdict['freq'])).astype(int)
                if argsdict['dim']==1:
                    v0_inviscid=revecs[:,ind]
                    w0_inviscid=np.conjugate(levecs[:,ind])
                    v=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)
                    w=np.zeros((2,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1)),dtype=np.complex128)
                    v[1,argsdict['Nt']+offset]=v0_inviscid
                    w[1,argsdict['Nt']+offset]=w0_inviscid
                    mat=viscid_mat1d

                elif argsdict['dim']==2:
                    v0_inviscid=revecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1)))
                    w0_inviscid=np.conjugate(levecs[:,ind].reshape(((2*argsdict['Nx']+1),(2*argsdict['Ny']+1))))
                    v=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
                    w=np.zeros((3,(2*argsdict['Nt']+1),(2*argsdict['Nx']+1),(2*argsdict['Ny']+1)),dtype=np.complex128)
                    v[2,argsdict['Nt']+offset]=v0_inviscid
                    w[2,argsdict['Nt']+offset]=w0_inviscid
                    mat=viscid_mat2d

                argsdict['As']=0
                omega=omega_inviscid
                omegas_i,vs_i,ws_i=rayleigh_mat(omega,v.ravel(),w.ravel(),mat,argsdict)
                print(omega_inviscid, omegas_i[-1])

                omegas=omegas+[omegas_i[-1]]
                vs=vs+[vs_i[-1]]
                ws=ws+[ws_i[-1]]

    else:
        argsdict_old=json.load(open(argsdict['filebase']+'argsdict.json'))
        if argsdict['dim']==1:
            mat=viscid_mat1d
        elif argsdict['dim']==2:
            mat=viscid_mat2d
        sweeps=argsdict_old['sweeps'].split(' ')
        sweep=0
        for sweep in range(len(sweeps)//3):
            par=sweeps[3*sweep]
            parf=float(sweeps[3*sweep+2])
            argsdict[par]=parf
        print(argsdict)
        omegans=np.load(argsdict['filebase']+"_"+str(sweep)+"_"+"evals.npy")
        vns,wns=np.load(argsdict['filebase']+"_"+str(sweep)+"_"+"evecs.npy")
        omegas=[omegan[-1] for omegan in omegans]
        vs=[vn[-1] for vn in vns]
        ws=[wn[-1] for wn in wns]
        argsdict['filebase']=argsdict['filebase']+'_cont'

        print(omegas)


    #sweep the substrate amplitude
    sweeps=argsdict['sweeps'].split(' ')
    for sweep in range(len(sweeps)//3):
        par=sweeps[3*sweep]
        pari=float(sweeps[3*sweep+1])
        parf=float(sweeps[3*sweep+2])
        print(sweep, par, pari, parf)
        argsdict['par']=par
        argsdict['pari']=pari
        argsdict['parf']=parf
        omegans,vns,wns,parns=cont(omegas, vs, ws, mat, argsdict)
        omegas=[omegan[-1] for omegan in omegans]
        vs=[vn[-1] for vn in vns]
        ws=[wn[-1] for wn in wns]
        print(omegas)
        np.save(argsdict['filebase']+"_"+str(sweep)+"_"+"pars.npy",np.array(parns))
        np.save(argsdict['filebase']+"_"+str(sweep)+"_"+"evals.npy",np.array(omegans))
        np.save(argsdict['filebase']+"_"+str(sweep)+"_"+"evecs.npy",[np.array(vns),np.array(wns)])
    jdump=json.dumps(argsdict)
    f=open(argsdict['filebase']+'argsdict.json','w')
    f.write(jdump)
    f.close()

    stop=timeit.default_timer()
    print('Runtime: ', stop-start)

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
        dim=1
        kx=0.45*np.pi
        ky=0
        itmax=10
        nmodes=2
        negative=1
        epsu=1e-8
        epsl=1e-8
        verbose=False
        def __str__(self):
            return str(self.__dict__)
    argsdict=dict(args.__dict__)
