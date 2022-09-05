#include "auto_f2c.h"
#define PI 3.14159265359

int func (integer ndim, const doublereal *w, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *f, doublereal *dfdu, doublereal *dfdp)
{
  /* System generated locals */
  integer dfdu_dim1 = ndim;
  integer dfdp_dim1 = ndim;
  double amp = par[0];
  double omega = par[1];
  double k= par[2];
  //system call to viscid.py

  if (ijac == 0) {
    return 0;
  }


  if (ijac == 1) {
    return 0;
  }

  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int stpnt (integer ndim, doublereal t,
           doublereal *u, doublereal *par)
{
  par[0] = 0.05;
  par[1] = 3.5;
  par[2]=0.2;
  int j, N = (ndim-2)/6;
  for (j=0; j<N; j++){
      u[0*N+j]=1.0;
      u[1*N+j]=0;
      u[2*N+j]=0;
      u[3*N+j]=1.0;
      u[4*N+j]=0;
      u[5*N+j]=0;
  }
  u[6*N]=1;
  u[6*N+1]=0;
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int pvls (integer ndim, const doublereal *u,
          doublereal *par)
{
  extern double getp();
  integer NDX  = getp("NDX", 0, u);
  integer u_dim1 = NDX;

  par[4]=getp("STA",0,u);
  return 0;
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int bcnd (integer ndim, const doublereal *par, const integer *icp,
          integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
          doublereal *fb, doublereal *dbc)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int icnd (integer ndim, const doublereal *par, const integer *icp,
          integer nint, const doublereal *u, const doublereal *uold,
          const doublereal *udot, const doublereal *upold, integer ijac,
          doublereal *fi, doublereal *dint)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
int fopt (integer ndim, const doublereal *u, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *fs, doublereal *dfdu, doublereal *dfdp)
{
  return 0;
}
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
