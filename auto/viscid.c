#include "auto_f2c.h"
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265359
int count=0;

int func (integer ndim, const doublereal *u, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *f, doublereal *dfdu, doublereal *dfdp)
{
  count=count+1;
  /* System generated locals */
  integer dfdu_dim1 = ndim;
  integer dfdp_dim1 = ndim;
  double amp = par[0];
  double freq = par[1];
  double k= par[2];
  printf("%i\t%e\t%e\t%e\t%e\t%e\n",count,amp,freq,k,u[ndim-2],u[ndim-1]);
  fflush(stdout);


  //write the state to autou.txt
  FILE *fileout;
  fileout = fopen("autou.txt","w");
  int result;
  for (int i=0; i<ndim; i++)
    result=fprintf(fileout,"%e\n", u[i]);
  fclose(fileout);

  //system call to viscid.py
  char str[512];
  sprintf(str, "./viscid_mat.py --filebase auto --acceleration %f --frequency %f --kx %f --strpnt 0", amp, freq, k);
  int call=system(str);

  //read the functions from to autof.txt
  FILE *filein;
  filein = fopen("autof.txt","r");
  for (int i=0; i<ndim; i++)
    result=fscanf(filein,"%lf", &f[i]);
  fclose(filein);

  if (ijac == 0) {
    return 0;
  }

  //read the jacobian from to autoJ.txt
  filein = fopen("autoJ.txt","r");
  for (int i=0; i<ndim; i++){
    for (int j=0; j<ndim; j++){
      doublereal temp;
      result=fscanf(filein,"%lf", &temp);
      ARRAY2D(dfdu,i,j)=temp;
    }
  }
  fclose(filein);

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
  par[0]=0;
  par[1]=5.0;
  par[2]=1.57079632679;
  double amp = par[0];
  double freq = par[1];
  double k= par[2];

  //system call to viscid.py
  char str[512];
  sprintf(str, "./viscid_mat.py --filebase auto --acceleration %f --frequency %f --kx %f --strpnt 1", amp, freq, k);
  fflush(stdout);
  int call=system(str);

  //read the state from to autou.txt
  FILE *file;
  int result;
  file = fopen("autou.txt","r");
  for (int i=0; i<ndim; i++)
    result=fscanf(file,"%lf", &u[i]);
  fclose(file);


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
