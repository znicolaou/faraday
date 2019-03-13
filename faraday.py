#!/usr/bin/env python
from __future__ import print_function
import os
import time
import argparse
import sys

#Command line arguments
parser = argparse.ArgumentParser(description='Moving mesh simulation for inviscid Faraday waves with inhomogeneous substrate.')
parser.add_argument("--filebase", type=str, required=True, dest='output', help='Base string for file output')
parser.add_argument("--frequency", type=float, default=23.0, dest='freq', help='Driving frequency in Hertz')
parser.add_argument("--gravity", type=float, default=980.0, dest='g', help='Gravitational acceleration in cm/s^2')
parser.add_argument("--acceleration", type=float, default=1.0, dest='acceleration', help='Driving acceleration in terms of gravitational acceleration')
parser.add_argument("--width", type=float, default=1.0, dest='width', help='Width in cm')
parser.add_argument("--length", type=float, default=4.0, dest='length', help='Length in cm')
parser.add_argument("--height", type=float, default=2.0, dest='height', help='Height in cm')
parser.add_argument("--radius", type=float, default=4.0, dest='radius', help='Radius in cm')
parser.add_argument("--tension", type=float, default=72.0,dest='sigma', help='Surface tension in dyne/cm^2')
parser.add_argument("--density", type=float, default=1.0,dest='rho', help='Fluid density in g/cm^3')
parser.add_argument("--time", type=float, default=50, dest='simTime', help='Simulation time in driving cycles')
parser.add_argument("--steps", type=float, default=15, dest='steps', help='Output steps per cycle')
parser.add_argument("--output", type=int, choices = [0, 1], default = 1, dest='outLevel', help='Flag to output full data or abbreviated data')
parser.add_argument("--iseed", type=int, default=0, dest='iseed', help='Seed for random initial conditions')
parser.add_argument("--iamp", type=float, default=1e-4, dest='iamp', help='Amplitude for modes in random initial conditions')
parser.add_argument("--imodes", type=int, default=10, dest='imodes', help='Number of modes to include in random initial conditions')
parser.add_argument("--sseed", type=int, default=0, dest='sseed', help='Seed for random substrate shape')
parser.add_argument("--samp", type=float, default=0.0, dest='samp', help='Amplitude for modes in random substrate shape')
parser.add_argument("--smodes", type=int, default=3, dest='smodes', help='Number of modes to include in random substrate shape')
parser.add_argument("--rtol", type=float, default=1e-4, dest='rtol', help='Integration relative tolerance')
parser.add_argument("--atol", type=float, default=1e-8, dest='atol', help='Integration absolute tolerance')
parser.add_argument("--damp1", type=float, default=2.0, dest='damp1', help='Constant damping coefficient')
parser.add_argument("--damp3", type=float, default=0.5, dest='damp2', help='Curvature damping coefficient')
parser.add_argument("--xmesh", type=int, default=10, dest='xmesh', help='Lateral mesh refinement')
parser.add_argument("--ymesh", type=int, default=5, dest='ymesh', help='Lateral mesh refinement')
parser.add_argument("--zmesh", type=int, default=5, dest='zmesh', help='Vertical mesh refinement')
parser.add_argument("--threshold", type=float, default=3.0, dest='thrs', help='Threshold change in log norm magnitude to stop integration')
parser.add_argument("--refinement", type=int, default=0, dest='refinement', help='Number of refinements for top')
parser.add_argument("--bmesh", type=int, choices = [0, 1], default = 1, dest='bmesh', help='Flag to move boundary mesh in time stepping. This is faster and tentatively more accurate than the alternative mesh movement, but suffers numerical instabilities for large deviations.')
parser.add_argument("--nonlinear", type=int, choices = [0, 1], default = 0, dest='nonlinear', help='Flag to include nonlinear terms')
parser.add_argument("--geometry", type=str, choices = ['rectangle', 'cylinder', 'box'], default = 'rectangle', dest='geometry', help='Mesh geometry. Options are rectangle, cylinder, and box.')
parser.add_argument("--contact", type=str, choices = ['stick', 'slip', 'periodic'], default = 'stick', dest='contact', help='Contact line boundary conditions. Options are stick, slip, and periodic. periodic is not available for cylinder geometry.')
parser.add_argument("--nthreads", type=int, default = 1, dest='nthreads', help='Number of threads to allow parallel computations to run over.')
args = parser.parse_args()

if(args.geometry=='cylinder' and args.contact=='periodic'):
	print('Periodic boundaries not supported for cylinders!')
	quit()

#We must set OMP_NUM_THREADS before importing packages to parallelize with the specified number of threads
os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
from dolfin import *
from mshr import Cylinder, generate_mesh
import numpy as np
from scipy.integrate import ode
from scipy.signal import argrelextrema
from scipy.stats import linregress
from scipy.interpolate import griddata
from scipy.special import jnp_zeros, jn_zeros, jn
set_log_level(30)


#Define timescales
omega=2.0*np.pi*args.freq
dt = 2.0*np.pi/omega/args.steps
tmax = 2.0*np.pi/omega*args.simTime
t_vec = np.arange((args.steps+1)*dt, tmax, dt)

#Set up mesh
tankHeight = args.height
if(args.geometry == 'rectangle'):
	dim=1 #dimension of the surface
	tankWidth = args.width
	meshHeight=tankHeight
	mesh=RectangleMesh(Point(0.0,0.0), Point(tankWidth,tankHeight),args.xmesh, args.zmesh, 'right/left')
elif (args.geometry == 'cylinder'):
	dim=2 #dimension of the surface
	tankRadius = args.radius
	meshHeight = args.zmesh/args.xmesh*tankRadius
	cylinder=Cylinder(Point(0.0,0.0,0.0),Point(0.0,0.0,meshHeight),tankRadius,tankRadius)
	mesh=generate_mesh(cylinder,args.xmesh)
elif (args.geometry == 'box'):
	dim=2 #dimension of the surface
	tankWidth = args.width
	tankLength = args.length
	meshHeight=tankHeight
	mesh=BoxMesh(Point(0.0,0.0), Point(tankLength,tankWidth,tankHeight),args.xmesh, args.ymesh,args.zmesh)

#Refine the top cells
def refine_top(mesh):
	topvertices = np.array(np.where(np.abs(mesh.coordinates()[:,dim]-meshHeight)< 10*DOLFIN_EPS)[0])
	topcells=[]
	for i in range(len(mesh.cells())):
		for vertex in mesh.cells()[i]:
			if(np.any(topvertices==vertex)):
				topcells.append(i)
	cell_markers=cpp.mesh.MeshFunctionBool(mesh,dim+1)
	for i in topcells:
		cell_markers.set_value(i,True)
	return refine(mesh,cell_markers)
for i in range(args.refinement):
	mesh=refine_top(mesh)

#indices of all top coordinates
idx_top2 = np.array(np.where(np.abs(mesh.coordinates()[:,dim]-meshHeight)< 10*DOLFIN_EPS)[0])
#pseudospectral method requires sorted indices for rectangles
if(args.geometry=='rectangle'):
	idx_top2=idx_top2[np.argsort(mesh.coordinates()[idx_top2,0])]
#estimate for the length between surface mesh points
meshlen=BoundaryMesh(mesh, "exterior", True).rmin()
#indices of mesh top coordinates not on contact line
if(args.geometry == 'rectangle'):
	idx_top = np.array(np.where([np.abs(coord[1]-meshHeight) < 10*DOLFIN_EPS and np.abs((coord[0]**2)**(0.5)-tankWidth) > 0.1*meshlen and np.abs((coord[0]**2)**(0.5)) > 0.1*meshlen for coord in mesh.coordinates()])[0])
elif(args.geometry == 'cylinder'):
	idx_top = np.array(np.where([np.abs(coord[2]-meshHeight) < 0.5*meshHeight/args.zmesh and np.abs((coord[0]**2+coord[1]**2)**(0.5)-tankRadius) > 0.5*meshlen for coord in mesh.coordinates()])[0])
elif(args.geometry == 'box'):
	idx_top = np.array(np.where([np.abs(coord[dim]-meshHeight) < 10*DOLFIN_EPS and np.abs((coord[0]**2)**(0.5)-tankLength) > 0.1*meshlen and np.abs((coord[0]**2)**(0.5)) > 0.1*meshlen and np.abs((coord[1]**2)**(0.5)-tankWidth) > 0.1*meshlen and np.abs((coord[1]**2)**(0.5)) > 0.1*meshlen for coord in mesh.coordinates()])[0])
if args.contact == 'stick':
	nt=len(idx_top)
else:
	nt=len(idx_top2)
#indices of top coordinates list that don't lie on the contact line
top2pos=np.array([np.argwhere(idx_top2==it)[0,0] for it in idx_top])
#indices of top coordinates that lie on the contact line
contactpos=np.array(np.argwhere([not np.any(idx_top==it) for it in idx_top2])[:,0])
#indices of bottom coordinates
idx_bottom = np.array(np.where(np.abs(mesh.coordinates()[:,dim]) < 10*DOLFIN_EPS)[0])
nb=len(idx_bottom)

#Initial surface height
y0 = np.zeros(2*nt, float)
t0=0
np.random.seed(args.iseed)
if (os.path.exists(args.output+"ic.dat")):
	print("Using initial conditions from file")
	ic=np.loadtxt(args.output+"ic.dat")
	t0=ic[0,0]
	vals=ic[0,1:]
	nt0=int(len(vals)/2)
	if(args.geometry == 'rectangle'):
		points=ic[1,1:nt0+1]
	else:
		points=np.transpose(np.array([ic[1,1:nt0+1],ic[1,nt0+1:]]))
	if (args.contact == 'stick'):
		if(args.geometry == 'rectangle'):
			y0[:nt]=griddata(points, vals[:nt0], mesh.coordinates()[idx_top,:dim], method='cubic')[:,0]
			y0[nt:]=griddata(points, vals[nt0:], mesh.coordinates()[idx_top,:dim], method='cubic')[:,0]
		else:
			y0[:nt]=griddata(points, vals[:nt0], mesh.coordinates()[idx_top,:dim], method='cubic')
			y0[nt:]=griddata(points, vals[nt0:], mesh.coordinates()[idx_top,:dim], method='cubic')
	elif (args.geometry == 'cylinder'):
		#For the cylinder, the mesh may contain points that lie outside the convex hull of the initial grid. Use nearest interpolation in this case
		y0[:nt]=griddata(points, vals[:nt0], mesh.coordinates()[idx_top2,:dim], method='nearest')
		y0[nt:]=griddata(points, vals[nt0:], mesh.coordinates()[idx_top2,:dim], method='nearest')
	else:
		if(args.geometry == 'rectangle'):
			y0[:nt]=griddata(points, vals[:nt0], mesh.coordinates()[idx_top2,:dim], method='cubic')[:,0]
			y0[nt:]=griddata(points, vals[nt0:], mesh.coordinates()[idx_top2,:dim], method='cubic')[:,0]
		else:
			y0[:nt]=griddata(points, vals[:nt0], mesh.coordinates()[idx_top2,:dim], method='cubic')
			y0[nt:]=griddata(points, vals[nt0:], mesh.coordinates()[idx_top2,:dim], method='cubic')
else:
	if(args.geometry=='rectangle'):
		isin=2*args.iamp*(np.random.random(args.imodes)-0.5)
		icos=2*args.iamp*(np.random.random(args.imodes)-0.5)
		for k in range(nt):
			X = mesh.coordinates()[idx_top2[k],0]
			val=0.0
			for n1 in range(1,args.imodes):
				if not args.contact == 'slip':
					val+=isin[n1]*np.sin(2*np.pi*n1*X/tankWidth)
			for n1 in range(1,args.imodes):
				if not args.contact == 'stick':
					val+=icos[n1]*np.cos(2*np.pi*n1*X/tankWidth)
			y0[k] = val
	elif(args.geometry=='cylinder'):
		isin=2*args.iamp*(np.random.random((args.imodes,args.imodes))-0.5)
		icos=2*args.iamp*(np.random.random((args.imodes,args.imodes))-0.5)
		for k in range(nt):
			X,Y = mesh.coordinates()[idx_top2[k],:dim]
			r=(X*X+Y*Y)**(0.5)
			theta=np.arctan2(Y,X)
			val=0.0
			for n1 in range(args.imodes):
				zeros=jnp_zeros(n1,args.imodes)
				for n2 in range(len(zeros)):
					val+=isin[n1,n2]*jn(n1,zeros[n2]*r/tankRadius)*np.sin(n1*theta)
			for n1 in range(args.imodes):
				zeros=jnp_zeros(n1,args.imodes)
				for n2 in range(len(zeros)):
					val+=icos[n1,n2]*jn(n1,zeros[n2]*r/tankRadius)*np.cos(n1*theta)
			y0[k] = val
	elif(args.geometry=='box'):
		iss=2*args.iamp*(np.random.random((args.imodes,args.imodes))-0.5)
		isc=2*args.iamp*(np.random.random((args.imodes,args.imodes))-0.5)
		ics=2*args.iamp*(np.random.random((args.imodes,args.imodes))-0.5)
		icc=2*args.iamp*(np.random.random((args.imodes,args.imodes))-0.5)
		for k in range(nt):
			X = mesh.coordinates()[idx_top2[k],0]
			Y = mesh.coordinates()[idx_top2[k],1]
			val=0.0
			for n1 in range(1,args.imodes):
				for n2 in range(1,args.imodes):
					val+=iss[n1,n2]*np.sin(np.pi*n1*X/tankLength)*np.sin(np.pi*n2*Y/tankWidth)
					val+=ics[n1,n2]*np.cos(np.pi*n1*X/tankLength)*np.sin(np.pi*n2*Y/tankWidth)
					val+=isc[n1,n2]*np.sin(np.pi*n1*X/tankLength)*np.cos(np.pi*n2*Y/tankWidth)
					val+=icc[n1,n2]*np.cos(np.pi*n1*X/tankLength)*np.cos(np.pi*n2*Y/tankWidth)
			y0[k] = val
#Substrate shape
h0 = np.zeros(nb, float)
np.random.seed(args.sseed)
if (os.path.exists(args.output+"substrate.dat")):
	print("Using substrate from files")
	substrateFile=open(args.output+"substrate.dat", 'r')
	if(args.geometry=='rectangle'):
		args.smodes=int(substrateFile.readline())
		ssin=args.samp*np.array([float(i) for i in substrateFile.readline().split()])
		scos=args.samp*np.array([float(i) for i in substrateFile.readline().split()])
	elif(args.geometry=='cylinder'):
		args.smodes=int(substrateFile.readline())
		ssin=args.samp*np.array([[float(i) for i in substrateFile.readline().split()] for j in range(args.smodes)])
		substrateFile.readline()
		scos=args.samp*np.array([[float(i) for i in substrateFile.readline().split()] for j in range(args.smodes)])
	elif(args.geometry=='box'):
		args.smodes=int(substrateFile.readline())
		subss=args.samp*np.array([[float(i) for i in substrateFile.readline().split()] for j in range(args.smodes)])
		substrateFile.readline()
		subsc=args.samp*np.array([[float(i) for i in substrateFile.readline().split()] for j in range(args.smodes)])
		substrateFile.readline()
		subcs=args.samp*np.array([[float(i) for i in substrateFile.readline().split()] for j in range(args.smodes)])
		substrateFile.readline()
		subcc=args.samp*np.array([[float(i) for i in substrateFile.readline().split()] for j in range(args.smodes)])
	substrateFile.close()
else:
	if(args.geometry=='rectangle'):
		ssin=2*args.samp*(np.random.random(args.smodes)-0.5)
		scos=2*args.samp*(np.random.random(args.smodes)-0.5)
	elif(args.geometry=='cylinder'):
		ssin=2*args.samp*(np.random.random((args.smodes,args.smodes))-0.5)
		scos=2*args.samp*(np.random.random((args.smodes,args.smodes))-0.5)
	elif(args.geometry=='box'):
		subss=2*args.samp*(np.random.random((args.smodes,args.smodes))-0.5)
		subsc=2*args.samp*(np.random.random((args.smodes,args.smodes))-0.5)
		subcs=2*args.samp*(np.random.random((args.smodes,args.smodes))-0.5)
		subcc=2*args.samp*(np.random.random((args.smodes,args.smodes))-0.5)
if(args.geometry=='rectangle'):
	for k in range(nb):
		X = mesh.coordinates()[idx_bottom[k],0]
		val=0.0
		for n1 in range(1,len(ssin)):
			val+=ssin[n1]*np.sin(2*np.pi*n1*X/tankWidth)
		for n1 in range(1,len(scos)):
			val+=scos[n1]*np.cos(2*np.pi*n1*X/tankWidth)
		h0[k] = val
elif(args.geometry=='cylinder'):
	for k in range(nb):
		X,Y = mesh.coordinates()[idx_bottom[k],:dim]
		r=(X*X+Y*Y)**(0.5)
		theta=np.arctan2(Y,X)
		val=0.0
		for n1 in range(len(ssin)):
			zeros=jn_zeros(n1,len(ssin)+1)
			for n2 in range(len(ssin[n1])):
				max=np.max([jn(n1,zeros[n2]*i/100) for i in range(101)])
				val+=ssin[n1,n2]*jn(n1,zeros[n2]*r/tankRadius)/max*np.sin(n1*theta)
		for n1 in range(len(scos)):
			zeros=jn_zeros(n1,len(scos)+1)
			for n2 in range(len(scos[n1])):
				max=np.max([jn(n1,zeros[n2]*i/100) for i in range(101)])
				val+=scos[n1,n2]*jn(n1,zeros[n2]*r/tankRadius)/max*np.cos(n1*theta)
		h0[k] = val
elif(args.geometry=='box'):
	for k in range(nb):
		X,Y = mesh.coordinates()[idx_bottom[k],:dim]
		val=0.0
		for n1 in range(args.smodes):
			for n2 in range(args.smodes):
				val+=subss[n1,n2]*np.sin(2*np.pi*(n1+1)*X/tankLength)*np.sin(2*np.pi*n2*Y/tankWidth)
				val+=subsc[n1,n2]*np.sin(2*np.pi*(n1+1)*X/tankLength)*np.cos(2*np.pi*n2*Y/tankWidth)
				val+=subcs[n1,n2]*np.cos(2*np.pi*(n1+1)*X/tankLength)*np.sin(2*np.pi*n2*Y/tankWidth)
				val+=subcc[n1,n2]*np.cos(2*np.pi*(n1+1)*X/tankLength)*np.cos(2*np.pi*n2*Y/tankWidth)
		h0[k] = val

#Move mesh functions
#Define displacement for every mesh point based on the top and bottom shapes
W=VectorFunctionSpace(mesh,"CG",1)
class meshDisplacement(UserExpression):
	def eval(self, values, x):
		zb=griddata(mesh.coordinates()[idx_bottom,:dim],mesh.coordinates()[idx_bottom, dim],np.array(x[:dim]),method='nearest')
		zt=griddata(mesh.coordinates()[idx_top2,:dim],mesh.coordinates()[idx_top2, dim],np.array(x[:dim]),method='nearest')
		nzb=griddata(mesh.coordinates()[idx_bottom,:dim],self.h0,np.array(x[:dim]),method='nearest')
		nzt=griddata(mesh.coordinates()[idx_top2,:dim],self.h1,np.array(x[:dim]),method='nearest')
		dztop=nzt-zt
		dzbottom=nzb-zb
		values[:dim]=np.zeros(dim)
		values[dim]=dzbottom+(dztop-dzbottom)*(x[dim]-zb)/(zt-zb)
displacement=meshDisplacement(element=W.ufl_element())
displacement.h1=np.zeros(len(idx_top2))+tankHeight
def movemesh(yn, h0):
	displacement.h0=h0
	if(args.contact == 'stick'):
		displacement.h1[top2pos]=tankHeight+yn[:nt]
	else:
		displacement.h1=tankHeight+yn[:nt]
	ALE.move(mesh,displacement)
movemesh(np.zeros(nt),h0)
#indices of top and bottom coordinates in the boundary mesh
bmesh = BoundaryMesh(mesh, "exterior", True)
if args.contact == 'stick':
	coords=mesh.coordinates()[idx_top]
else:
	coords=mesh.coordinates()[idx_top2]
bidx_top=[]
for n in range(nt):
	bidx=np.array(np.where([(coords[n,:dim]==coord[:dim]).all() and np.abs(coord[dim]-tankHeight)<meshlen for coord in bmesh.coordinates()]))[0,0]
	bidx_top.append(bidx)
bidx_top=np.array(bidx_top)

def movebmesh(yn):
	bmesh.coordinates()[bidx_top,dim] = yn[:nt]+tankHeight
	ALE.move(mesh,bmesh)

#Boundary conditions for top surface
class DynamicFreeSurfaceCondition(UserExpression):
	def eval(self, values, x):
		if args.contact == 'stick':
			values[0]=griddata(mesh.coordinates()[idx_top,:dim], self.phi_on_surface, np.array(x[:dim]), method='linear')[0]
		else:
			values[0]=griddata(mesh.coordinates()[idx_top2,:dim], self.phi_on_surface, np.array(x[:dim]), method='linear')[0]
def boundary_T(x):
	if args.contact == 'stick':
		return np.any(np.linalg.norm(mesh.coordinates()[idx_top]-x, axis=1)<10*DOLFIN_EPS)
	else:
		return np.any(np.linalg.norm(mesh.coordinates()[idx_top2]-x, axis=1)<10*DOLFIN_EPS)
class PeriodicBoundary(SubDomain):
	def inside(self, x, on_boundary):
		if(args.geometry=='rectangle'):
			return bool(x[0] == 0.0)
		elif(args.geometry=='box'):
			return bool(x[0] == 0.0 or x[1] == 0.0)
	def map(self, x, y):
		if(args.geometry=='rectangle'):
			y[0] = x[0] - tankWidth
			y[1] = x[1]
		elif(args.geometry=='box'):
			y[0] = x[0] - tankLength
			y[1] = x[1] - tankWidth
			y[2] = x[2]

# Output before integration
if args.outLevel == 1:
	paramOut = open(args.output+".txt",'w')
	paramOut.write(" ".join(sys.argv)+"\n")
	paramOut.write("Steps\n%f\n" % args.steps)
	paramOut.write("Height\n%f\n" % tankHeight)
	if args.geometry=='rectangle':
		paramOut.write("Width\n%f\n" % tankWidth)
	if args.geometry=='cylinder':
		paramOut.write("Radius\n%f\n" % tankRadius)
	if args.geometry=='box':
		paramOut.write("Width\n%f\n" % tankWidth)
		paramOut.write("Length\n%f\n" % tankLength)
	paramOut.write("Mesh points\n")
	paramOut.write("%i\n" % len(mesh.coordinates()))
	paramOut.write("Surface points coordinates\n")
	for i in range(len(idx_top2)):
		print(idx_top2[i], file=paramOut, end=" ")
	print('', file=paramOut)
	paramOut.write("Interior surface points coordinates\n")
	for i in range(len(idx_top)):
		print(idx_top[i], file=paramOut, end=" ")
	print('', file=paramOut)
	paramOut.write("Substrate points coordinates\n")
	for i in range(nb):
		print(idx_bottom[i], file=paramOut, end=" ")
	print('', file=paramOut)
	paramOut.write("Mesh cells\n")
	for i in range(len(mesh.cells())):
		for j in range(len(mesh.cells()[i])):
			print(mesh.cells()[i,j],file=paramOut, end=" ")
	print('', file=paramOut)
	paramOut.write("Results:\n")
	paramOut.close()
paramOut = open(args.output+".txt",'a+')
paramOut.write("%f %f " % (args.freq, args.acceleration))
paramOut.close()

#Time derivative function
if(args.contact == 'periodic'):
	V = FunctionSpace(mesh, "CG", 2,constrained_domain=PeriodicBoundary())
	U = FunctionSpace(mesh, "CG", 1,constrained_domain=PeriodicBoundary())
else:
	V = FunctionSpace(mesh, "CG", 2)
	U = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
phi = Function(V)
f = Expression("0.0", degree=0)
L = f*v*dx
bc_exp_top = DynamicFreeSurfaceCondition(element=V.ufl_element())
bc_top = DirichletBC(V, bc_exp_top, boundary_T, method="pointwise", check_midpoint=False)
dydt = np.zeros(nt*2, float)

#Function for finding curvature away from the contact line
#Interpolate the mesh to find finite differences, using spacing one tenth the mesh lengh scale
delta=0.1*meshlen
points=[]
if(dim == 2):
	if args.contact == 'stick':
		for k in range(len(idx_top2)):
			x=mesh.coordinates()[idx_top2[k],:dim]
			points.append(x)
			points.append(x+[delta,0.0])
			points.append(x-[delta,0.0])
			points.append(x+[0.0,delta])
			points.append(x-[0.0,delta])
			points.append(x+[delta,delta])
			points.append(x-[delta,delta])
	elif args.contact == 'slip':
		for k in range(len(idx_top2)):
			x=mesh.coordinates()[idx_top2[k],:dim]
			if not k in contactpos:
				points.append(x)
				points.append(x+[delta,0.0])
				points.append(x-[delta,0.0])
				points.append(x+[0.0,delta])
				points.append(x-[0.0,delta])
				points.append(x+[delta,delta])
				points.append(x-[delta,delta])
			elif args.geometry == 'cylinder':
				#On the contact line, rotate the stencil so that points remain in the convex hull.  The curvature and  dertivative products should be invariant.
				theta=np.arctan2(x[0],x[1])-np.pi/4
				rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
				points.append(x+np.dot(rot,[-delta,-delta]))
				points.append(x+np.dot(rot,[0,-delta]))
				points.append(x+np.dot(rot,[-2*delta,-delta]))
				points.append(x+np.dot(rot,[-delta,0]))
				points.append(x+np.dot(rot,[-delta,-2*delta]))
				points.append(x)
				points.append(x+np.dot(rot,[-2*delta,-2*delta]))
			elif args.geometry == 'box':
				#On the contact line, rotate the stencil so that points remain in the convex hull.  The curvature and  dertivative products should be invariant.
				theta=0
				if x[0]==0 and x[1]==0:
					theta=np.pi
				elif x[0]==tankLength and x[1]==0:
					theta=np.pi/2
				elif x[0]==tankLength and x[1]==tankWidth:
					theta=0
				elif x[0]==0 and x[1]==tankWidth:
					theta=-np.pi/2
				elif x[0]==0:
					theta=-np.pi/2
				elif x[0]==tankLength:
					theta=np.pi/2
				elif x[1]==0:
					theta=np.pi
				elif x[1]==tankWidth:
					theta=0
				rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
				points.append(x+np.dot(rot,[-delta,-delta]))
				points.append(x+np.dot(rot,[0,-delta]))
				points.append(x+np.dot(rot,[-2*delta,-delta]))
				points.append(x+np.dot(rot,[-delta,0]))
				points.append(x+np.dot(rot,[-delta,-2*delta]))
				points.append(x)
				points.append(x+np.dot(rot,[-2*delta,-2*delta]))
	elif args.contact == 'periodic':
		for k in range(len(idx_top2)):
			x=mesh.coordinates()[idx_top2[k],:dim]
			points.append(x)
			points.append(np.array([np.mod(x[0]+delta, tankLength),x[1]]))
			points.append(np.array([np.mod(x[0]-delta, tankLength),x[1]]))
			points.append(np.array([x[0],np.mod(x[1]+delta, tankWidth)]))
			points.append(np.array([x[0],np.mod(x[1]-delta, tankWidth)]))
			points.append(np.array([np.mod(x[0]+delta, tankLength),np.mod(x[1]+delta, tankWidth)]))
			points.append(np.array([np.mod(x[0]-delta, tankLength),np.mod(x[1]-delta, tankWidth)]))
else:
	if args.contact == 'stick':
		for k in range(len(idx_top)):
			x=mesh.coordinates()[idx_top[k],:dim]
			points.append(x)
			points.append(x+delta)
			points.append(x-delta)
	elif args.contact == 'slip':
		for k in range(len(idx_top2)):
			x=mesh.coordinates()[idx_top2[k],:dim]
			if not k in contactpos:
				points.append(x)
				points.append(x+delta)
				points.append(x-delta)
			else:
				if x[0] == 0:
					points.append(x+delta)
					points.append(x+2*delta)
					points.append(x)
				if x[0] == tankWidth:
					points.append(x-delta)
					points.append(x)
					points.append(x-2*delta)
	elif args.contact == 'periodic':
		frequencies = np.concatenate((np.arange(0,int(nt/2),1),1+np.arange(int(nt/2),nt-1,1)-nt))*2*np.pi/tankWidth
def curvature_top(y):
	if(dim == 1):
		if(args.contact == 'stick'):
			#Interpolate the mesh and use finite differnces
			mvals=mesh.coordinates()[idx_top2,1]
			mvals[top2pos]=tankHeight+y[:nt]
			vals=griddata(mesh.coordinates()[idx_top2,0], mvals, np.array(points), method='cubic')
			hx=(vals[1::3]-vals[2::3])/(2*delta)
			hxx=(vals[1::3]+vals[2::3]-2*vals[0::3])/(delta*delta)
			if(args.nonlinear==1):
				curve=hxx/(1+hx*hx)**(1.5)
			else:
				curve=hxx
			ret=[hx[:,0],curve[:,0]]
		elif(args.contact == 'periodic'):
			#Pseudospectral approach with fft
			hx = np.real(np.fft.ifft(1j*frequencies*np.fft.fft(y[:nt-1])))
			hxx = -np.real(np.fft.ifft(frequencies**2*np.fft.fft(y[:nt-1])))
			if(args.nonlinear==1):
				curve0=hxx/(1+hx*hx)**1.5
			else:
				curve0=hxx
			ret=[np.append(hx,hx[0]),np.append(curve0,curve0[0])]
		elif args.contact == 'slip':
				vals=griddata(mesh.coordinates()[idx_top2,:dim], tankHeight+y[:nt], np.array(points), method='cubic')
				hx=(vals[1::3]-vals[2::3])/(2*delta)
				hxx=(vals[1::3]+vals[2::3]-2*vals[0::3])/(delta*delta)
				if(args.nonlinear == 1):
					curve2=hxx/(1+hx*hx)**(1.5)
				else:
					curve2=hxx
				curve2[contactpos]/=2 #this is a numerical-empirical contact line slip force adjustment that is stable. Maybe it makes sense physically.
				ret=[hx[:,0],curve2[:,0]]
	if(dim == 2):
		if args.contact == 'slip':
			vals=griddata(mesh.coordinates()[idx_top2,:dim], tankHeight+y[:nt], np.array(points), method='cubic')
			# nans=np.floor(np.array(np.where(np.isnan(vals))[0])/7)
			# print()
			# print(len(nans),len(contactpos))
			# print(nans)
			# print(contactpos)
			# quit()
			hx=(vals[1::7]-vals[2::7])/(2*delta)
			hy=(vals[3::7]-vals[4::7])/(2*delta)
			hxx=(vals[1::7]+vals[2::7]-2*vals[0::7])/(delta*delta)
			hyy=(vals[3::7]+vals[4::7]-2*vals[0::7])/(delta*delta)
			hxy=(vals[5::7]+vals[6::7]-vals[1::7]-vals[2::7]-vals[3::7]-vals[4::7]+2*vals[0::7])/(2*delta*delta)
			if(args.nonlinear == 1):
				curve2=(hxx+hyy+hxx*hy*hy+hyy*hx*hx-2*hx*hy*hxy)/(1+hx*hx+hy*hy)**(1.5)
			else:
				curve2=hxx+hyy
			curve2[contactpos]/=2 #this is a numerical-empirical contact line slip force adjustment that is stable. Maybe it makes sense physically.
			ret=[hx,hy,curve2]
		elif args.contact == 'stick':
			mvals=mesh.coordinates()[idx_top2,dim]
			mvals[top2pos]=tankHeight+y[:nt]
			vals=griddata(mesh.coordinates()[idx_top2,:dim], mvals, np.array(points), method='cubic')
			hx=(vals[1::7]-vals[2::7])/(2*delta)
			hy=(vals[3::7]-vals[4::7])/(2*delta)
			hxx=(vals[1::7]+vals[2::7]-2*vals[0::7])/(delta*delta)
			hyy=(vals[3::7]+vals[4::7]-2*vals[0::7])/(delta*delta)
			hxy=(vals[5::7]+vals[6::7]-vals[1::7]-vals[2::7]-vals[3::7]-vals[4::7]+2*vals[0::7])/(2*delta*delta)
			if(args.nonlinear == 1):
				curve=(hxx+hyy+hxx*hy*hy+hyy*hx*hx-2*hx*hy*hxy)/(1+hx*hx+hy*hy)**(1.5)
			else:
				curve=hxx+hyy
			ret=[hx,hy,curve]
		elif args.contact == 'periodic':
				#A pseudospectral derivative may be possible here, but the refined meshes are not strictly square and the indices would have to be ordered. Use finite differences for simplicity.
				vals=griddata(mesh.coordinates()[idx_top2,:dim], tankHeight+y[:nt], np.array(points), method='cubic')
				hx=(vals[1::7]-vals[2::7])/(2*delta)
				hy=(vals[3::7]-vals[4::7])/(2*delta)
				hxx=(vals[1::7]+vals[2::7]-2*vals[0::7])/(delta*delta)
				hyy=(vals[3::7]+vals[4::7]-2*vals[0::7])/(delta*delta)
				hxy=(vals[5::7]+vals[6::7]-vals[1::7]-vals[2::7]-vals[3::7]-vals[4::7]+2*vals[0::7])/(2*delta*delta)
				if(args.nonlinear == 1):
					curve=(hxx+hyy+hxx*hy*hy+hyy*hx*hx-2*hx*hy*hxy)/(1+hx*hx+hy*hy)**(1.5)
				else:
					curve=hxx+hyy
				ret=[hx,hy,curve]
	return ret
#Function to return the time derivative
#We scale the velocity potential so the components of y have comparable scales
scale=args.g*2*np.pi/omega
def ode_deriv(t, y):
	#Move the bmesh every evaluation if the flag is set, otherwise move mesh between outputs
	if(args.bmesh==1):
		movebmesh(y)
	bc_exp_top.phi_on_surface=scale*y[nt:]
	solve(a == L, phi, bc_top)
	if dim == 2:
		[hx,hy,curve]=curvature_top(y)
	else:
		[hx, curve]=curvature_top(y)
	if args.contact == 'stick':
		phix = project(Dx(phi,0),U).compute_vertex_values()[idx_top]
		if dim == 2:
			phiy = project(Dx(phi,1),U).compute_vertex_values()[idx_top]
			phiz = project(Dx(phi,2),U).compute_vertex_values()[idx_top]
		else:
			phiz = project(Dx(phi,1),U).compute_vertex_values()[idx_top]
	else:
		phix = project(Dx(phi,0),U).compute_vertex_values()[idx_top2]
		if dim == 2:
			phiy = project(Dx(phi,1),U).compute_vertex_values()[idx_top2]
			phiz = project(Dx(phi,2),U).compute_vertex_values()[idx_top2]
		else:
			phiz = project(Dx(phi,1),U).compute_vertex_values()[idx_top2]

	dydt[:nt] = phiz-args.damp1*y[:nt]+args.damp2*curve
	dydt[nt:] = (args.sigma/args.rho*curve+args.g*(-1 + args.acceleration*sin(omega*(t+t0)))*y[:nt])/scale
	if args.nonlinear==1:
		if dim==1:
			dydt[:nt] += -phix*hx
			dydt[nt:] += (-0.5*(phix*phix+phiz*phiz) + phiz*dydt[:nt])/scale
		elif dim==2:
			dydt[:nt] += -phix*hx-phiy*hy
			dydt[nt:] += (-0.5*(phix*phix+phiy*phiy+phiz*phiz) + phiz*dydt[:nt])/scale
	return dydt

# Time integration
ode_integrator = ode(ode_deriv)
ode_integrator.set_initial_value(y0)
ode_integrator.set_integrator('vode',atol=args.atol, rtol=args.rtol, max_step=0.1*dt)
t1 = time.time()
itlast=0
if(dim==2):
	meshes=np.zeros((len(t_vec), len(mesh.coordinates()),3), float)
else:
	meshes=np.zeros((len(t_vec), len(mesh.coordinates()),2), float)
norms1=np.zeros((len(t_vec)),float)
norms2=np.zeros((len(t_vec)),float)
if(args.geometry=='cylinder'):
	projections=np.zeros((len(t_vec),args.imodes,2*args.imodes))
elif(args.geometry=='box'):
	projections=np.zeros((len(t_vec),args.imodes,args.imodes))
else:
	projections=np.zeros((len(t_vec),args.imodes))

#integrate one cycle before initializing the norm
if(args.bmesh==0):
	movemesh(y0,h0)
if(args.outLevel==1):
	print("%i surface points"%nt)
	print("Integrating one cycle: ", end=' ')
	sys.stdout.flush()
for i in range(1,args.steps):
	if(args.outLevel==1):
		print(i, end=' ')
		sys.stdout.flush()
	y = ode_integrator.integrate(i*dt)
	if(args.bmesh==0):
		movemesh(y,h0)
print('')
initnorm=np.linalg.norm(mesh.coordinates()[idx_top2,dim]-tankHeight) + 2*np.linalg.norm(ode_deriv(args.steps*dt,y)[:nt])/omega
for it, t in enumerate(t_vec):
	y = ode_integrator.integrate(t)
	if(args.bmesh==0):
		movemesh(y,h0)
	#Record mesh for output
	meshes[it] = np.array(mesh.coordinates())
	#Calculate norms for frequency estimation
	norms1[it] = np.linalg.norm(y[:nt])
	norms2[it] = 2*np.linalg.norm(ode_deriv(t,y)[:nt])/omega
	#Calculate the projections onto modes for wavenumber estimation
	projss=None #array of sine-sine mode projections
	projsc=None #array of sine-cosine mode projections
	projcs=None #array of cosine-sine mode projections
	projcc=None #array of cosine-cosine mode projections
	if args.geometry == 'rectangle':
		projss=np.zeros(args.imodes)
		projcc=np.zeros(args.imodes)
		projsc=np.zeros(args.imodes) #not for rectangles
		projcs=np.zeros(args.imodes) #not for rectangles
		for n1 in range(args.imodes):
			for k in range(nt):
				x,h = mesh.coordinates()[idx_top2[k]]
				# h = mesh.coordinates()[idx_top2[k],1]
				projcc[n1] += (h-tankHeight)*np.cos(np.pi*n1*x/tankWidth)
				projss[n1] += (h-tankHeight)*np.sin(np.pi*n1*x/tankWidth)
	elif(args.geometry == 'cylinder'):
		projss=np.zeros((args.imodes,2*args.imodes))
		projcc=np.zeros((args.imodes,2*args.imodes))
		projsc=np.zeros((args.imodes,2*args.imodes)) #not for cylinders
		projcs=np.zeros((args.imodes,2*args.imodes)) #not for cylinders
		for n1 in range(args.imodes):
			#modes with extrema at contact line
			zeros=jnp_zeros(n1,args.imodes)
			for n2 in range(args.imodes):
				for k in range(nt):
					X,Y,h = mesh.coordinates()[idx_top2[k]]
					r=(X*X+Y*Y)**(0.5)
					theta=np.arctan2(Y,X)
					projss[n1,n2] += r*(h-tankHeight)*jn(n1,zeros[n2]*r/tankRadius)*np.sin(n1*theta)
					projcc[n1,n2] += r*(h-tankHeight)*jn(n1,zeros[n2]*r/tankRadius)*np.cos(n1*theta)
			#modes with zeros at contact line
			zeros2=jn_zeros(n1,args.imodes)
			for n2 in range(args.imodes):
				for k in range(nt):
					X,Y,h = mesh.coordinates()[idx_top2[k]]
					r=(X*X+Y*Y)**(0.5)
					theta=np.arctan2(Y,X)
					projss[n1,args.imodes+n2] += r*(h-tankHeight)*jn(n1,zeros2[n2]*r/tankRadius)*np.sin(n1*theta)
					projcc[n1,args.imodes+n2] += r*(h-tankHeight)*jn(n1,zeros2[n2]*r/tankRadius)*np.cos(n1*theta)
	elif args.geometry == 'box':
		projss=np.zeros((args.imodes,args.imodes))
		projsc=np.zeros((args.imodes,args.imodes))
		projcs=np.zeros((args.imodes,args.imodes))
		projcc=np.zeros((args.imodes,args.imodes))
		for n1 in range(args.imodes):
			for n2 in range(args.imodes):
				for k in range(nt):
					X,Y,h = mesh.coordinates()[idx_top2[k]]
					projcc[n1,n2] += (h-tankHeight)*np.cos(np.pi*n1*X/tankLength)*np.cos(np.pi*n2*Y/tankWidth)
					projss[n1,n2] += (h-tankHeight)*np.sin(np.pi*n1*X/tankLength)*np.sin(np.pi*n2*Y/tankWidth)
					projsc[n1,n2] += (h-tankHeight)*np.sin(np.pi*n1*X/tankLength)*np.cos(np.pi*n2*Y/tankWidth)
					projcs[n1,n2] += (h-tankHeight)*np.cos(np.pi*n1*X/tankLength)*np.sin(np.pi*n2*Y/tankWidth)
	projections[it]=projss*projss+projsc*projsc+projcs*projcs+projcc*projcc #array of mode projections.

	#Output while integrating
	if args.outLevel == 1:
		print('%f\t%f\t%f\t%f\t%f' % (t/tmax, (norms1[it]+norms2[it])/initnorm, np.linalg.norm(y[:nt])/np.linalg.norm(y[nt:]), time.time()-t1, (time.time()-t1)*(tmax-t+dt)/(t+dt)))
		np.save(args.output,meshes[:it+1])
		np.save(args.output+"norms", (norms1[:it+1]+norms2[:it+1])/initnorm)
		if(args.contact == 'stick'):
			nt2=len(idx_top2)
			fs=np.zeros((2,2*nt2+1))
			fs[0,0]=t_vec[itlast]
			fs[0,top2pos+1]=y[:nt]
			fs[0,top2pos+1+nt2]=y[nt:]
			fs[1,1:nt2+1]=mesh.coordinates()[idx_top2,0]
			fs[1,nt2+1:]=mesh.coordinates()[idx_top2,0] #redundant data for rectangle geometry
			np.savetxt(args.output+"fs.dat",fs)
		else:
			fs=np.zeros((2,2*nt+1))
			fs[0,0]=t_vec[itlast]
			fs[0,1:nt+1]=y[:nt]
			fs[0,nt+1:]=y[nt:]
			fs[1,1:nt+1]=mesh.coordinates()[idx_top2,0]
			fs[1,nt+1:]=mesh.coordinates()[idx_top2,1] #redundant data for rectangle geometry
			np.savetxt(args.output+"fs.dat",fs)
		sys.stdout.flush()
	#Stop if the norm has changed by more than threshold
	itlast=it
	if np.abs(np.log((norms1[it]+norms2[it])/initnorm)/np.log(10)) > args.thrs:
		break


#Fit the growth rate
norms=(norms1[int(itlast/2):itlast+1]+norms2[int(itlast/2):itlast+1])/initnorm
T1=argrelextrema(norms1[int(itlast/2):], np.greater)[0]
rate, intercept, r_value, p_value, std_err = linregress(dt*np.arange(len(norms)),np.log(norms))
frequency=1
if(len(T1)>2):
	frequency=(args.steps/(np.mean(np.diff(T1))*2))
#Estimate the maximum mode projection. The wavenumber from this mode is outputed.
proj=np.mean(projections[int(itlast/2):itlast+1],axis=0)
if(args.geometry == 'rectangle'):
	wave=np.pi*np.argmax(proj)/tankWidth
if(args.geometry == 'cylinder'):
	[wavetheta,waver]=np.where(proj==np.max(proj))
	if(waver < args.imodes):
			wave=jnp_zeros(wavetheta[0],waver[0]+1)[-1]/tankRadius
	else:
			wave=jn_zeros(wavetheta[0],waver[0]-args.imodes+1)[-1]/tankRadius
elif (args.geometry == 'box'):
	[wavex,wavey]=np.array(np.where(proj==np.max(proj)))[:,0]
	wave=np.pi*wavex/tankLength+np.pi*wavey/tankWidth


#Output results after integration
paramOut = open(args.output+".txt",'a+')
paramOut.write("%f %f %f\n" % (frequency, rate, wave))
paramOut.close()
if(args.outLevel == 1):
	if(args.contact == 'stick'):
		nt2=len(idx_top2)
		fs=np.zeros((2,2*nt2+1))
		fs[0,0]=t_vec[itlast]
		fs[0,top2pos+1]=y[:nt]
		fs[0,top2pos+1+nt2]=y[nt:]
		fs[1,1:nt2+1]=mesh.coordinates()[idx_top2,0]
		fs[1,nt2+1:]=mesh.coordinates()[idx_top2,1]
		np.savetxt(args.output+"fs.dat",fs)
	else:
		fs=np.zeros((2,2*nt+1))
		fs[0,0]=t_vec[itlast]
		fs[0,1:nt+1]=y[:nt]
		fs[0,nt+1:]=y[nt:]
		fs[1,1:nt+1]=mesh.coordinates()[idx_top2,0]
		fs[1,nt+1:]=mesh.coordinates()[idx_top2,1]
		np.savetxt(args.output+"fs.dat",fs)

print("%f %f %f %f %f" % (args.freq, args.acceleration, frequency, rate, wave))
print("runtime %.2f seconds" % (time.time() - t1))

sys.stdout.flush()
