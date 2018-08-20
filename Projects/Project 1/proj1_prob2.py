#Imports
import numpy as np
import matplotlib.pyplot as pl
import math as m

#Project 1 Problem 2
#Chung Yin Leung

## Part A Contour Plots

#Read field line data
x = np.genfromtxt('data/naca0012_a8_x.dat', delimiter=',')
z = np.genfromtxt('data/naca0012_a8_z.dat', delimiter=',')
Cp = np.genfromtxt('data/naca0012_a8_Cp.dat', delimiter=',')

pl.figure()
pl.contour(x,z,Cp,750)
pl.colorbar()
pl.title('NACA 0012 Pressure Field at 8$ \degree $ $ \\alpha $')
pl.xlabel('x')
pl.ylabel('z')


## Part B Surface Pressure Plots & Calculations

#Define thickness distribution equation
def zt(xc):
	t = 0.12 #thickness ratio
	c = 1 #chord length
	zt = (t/0.2)*c*(0.2969*(xc ** (0.5)) - 0.1260*(xc) - 0.3516*(xc ** 2) + 0.2843*(xc ** 3) - 0.1015*(xc ** 4))
	return zt

#Define camber line equations of NACA 2412 based on chordwise position
def zc(xc):
	m = 0.02 #max camber ratio
	p = 0.4 #max camber chordwise position
	c = 1 #chord length
	#Equation piecewise definition using an If-else statement
	if xc < p:
		zc = (m/(p ** 2))*c*xc*(2*p - xc)
	else:
		zc = (m/((1-p) ** 2))*(1 - xc)*c*(1 + xc - 2*p)
	return zc

#Function for the derivative of thickness function
def dzt_dx(xc):
	t = 0.12 #thickness ratio
	c = 1 #chord length
	dzt_dx = (t/0.2)*(0.5*0.2969*(xc ** (-0.5)) - 0.1260 - 0.3516*2*xc + 0.2843*3*(xc ** 2) - 0.1015*4*(xc ** 3))
	return dzt_dx

#Function for the derivative of camber function
def dzc_dx(xc):
	m = 0.02 #max camber ratio
	p = 0.4 #max camber chordwise position
	c = 1 #chord length
	#Equation piecewise definition using an If-else statement
	if xc < p:
		dzc_dx = (2*m/(p ** 2))*(p - xc)
	else:
		dzc_dx = (2*m/((1-p) ** 2))*(p - xc)
	return dzc_dx

#Function for integrating normal force coefficient (Anderson Fund. 1.15)
#Lower and Upper terms are separated into own integral/integrand and integrated
#Integration by trapezoidal rule
#Assumes all friction coeff are 0
def Cn(Cp_l, x_l, Cp_u, x_u):
	c = 1; #chord length

	#1st term from 1.15 with only upper part
	integrand_l = [Cp_l[i] for i in range(0,len(Cp_l))]
	#Variable to store the sum of areas
	integrate_l = 0
	#Evaluates trapezoidal area between two x-values and adds to integration variable
	for i in range(0,len(Cp_l) - 1):
		integrate_l = integrate_l + 0.5*(integrand_l[i] + integrand_l[i+1])*(x_l[i+1] - x_l[i])

    #1st term from 1.15 with lower upper part
	integrand_u = [-Cp_u[i] for i in range(0,len(Cp_u))]

	integrate_u = 0
	for i in range(0,len(Cp_u) - 1):
		integrate_u = integrate_u + 0.5*(integrand_u[i] + integrand_u[i+1])*(x_u[i+1] - x_u[i])

	#Sum of all integrations
	Cn = (1/c)*(integrate_l + integrate_u)
	return Cn

#Function for integrating axial force coefficient (Anderson Fund. 1.16)
#Lower and Upper terms are separated into own integral/integrand and integrated
#Integration by trapezoidal rule
#Assumes all friction coeff are 0
def Ca(Cp_l, x_l, dyldx, Cp_u, x_u, dyudx):
	c = 1; #chord length

	#1st term from 1.16 with only upper part
	integrand_u = [Cp_u[i]*dyudx[i] for i in range(0,len(Cp_u))]

	integrate_u = 0
	for i in range(0,len(Cp_u) - 1):
		integrate_u = integrate_u + 0.5*(integrand_u[i] + integrand_u[i+1])*(x_u[i+1] - x_u[i])

	#1st term from 1.16 with only lower part
	integrand_l = [-Cp_l[i]*dyldx[i] for i in range(0,len(Cp_l))]

	integrate_l = 0
	for i in range(0,len(Cp_l) - 1):
		integrate_l = integrate_l + 0.5*(integrand_l[i] + integrand_l[i+1])*(x_l[i+1] - x_l[i])


	Ca = (1/c)*(integrate_u + integrate_l)
	return Ca

#Function for integrating normal force coefficient (Anderson Fund. 1.17)
#Lower and Upper terms are separated into own integral/integrand and integrated
#Integration by trapezoidal rule
#Assumes all friction coeff are 0
def CM(Cp_l, x_l, dyldx, yl, Cp_u, x_u, dyudx, yu):
	c = 1;

    #1st term from 1.17 with only upper part
	integrand1_u = [x_u[i]*(Cp_u[i]) for i in range(0,len(Cp_u))]
	integrate1_u = 0
	for i in range(0,len(Cp_u) - 1):
		integrate1_u = integrate1_u + 0.5*(integrand1_u[i] + integrand1_u[i+1])*(x_u[i+1] - x_u[i])

    #1st term from 1.17 with only lower part
	integrand1_l = [x_l[i]*(-Cp_l[i]) for i in range(0,len(Cp_l))]
	integrate1_l = 0
	for i in range(0,len(Cp_l) - 1):
		integrate1_l = integrate1_l + 0.5*(integrand1_l[i] + integrand1_l[i+1])*(x_u[i+1] - x_u[i])

    #3rd term from 1.17
	integrand2 = [Cp_u[i]*dyudx[i]*yu[i] for i in range(0,len(Cp_u))]
	integrate2 = 0
	for i in range(0,len(Cp_u) - 1):
		integrate2 = integrate2 + 0.5*(integrand2[i] + integrand2[i+1])*(x_u[i+1] - x_u[i])

    #4th term from 1.17
	integrand3 = [-Cp_l[i]*dyldx[i]*yl[i] for i in range(0,len(Cp_l))]
	integrate3 = 0
	for i in range(0,len(Cp_l) - 1):
		integrate3 = integrate3 + 0.5*(integrand3[i] + integrand3[i+1])*(x_l[i+1] - x_l[i])


	CM = (1/(c ** 2))*(integrate1_u + integrate1_l + integrate2 + integrate3)
	return CM

#Function that calculates/returns lift & drag coeff. using normal & axial coeff.
#Uses Anderson Fund. 1.18 & 1.19 for lift and drag, respectively.
def aero_C(cn, ca, alpha):
	cl1 = cn*m.cos(alpha*m.pi/180) - ca*m.sin(alpha*m.pi/180)
	cd1 = cn*m.sin(alpha*m.pi/180) + ca*m.cos(alpha*m.pi/180)
	return cl1, cd1

#Function that calculates center of pressure
#Uses equation with normal force (Anderson Fund. 1.20)
#Simplified to a ratio of coefficients.
def xcp(cm, cn):
	c = 1 #Chord length
	xcp = -1*(cm/cn)*c
	return xcp

#NACA 0012 AOA 0 deg
#Retrieve data and stores x-values and the surface pressure values
x1, SCp1_0 = np.genfromtxt('data/naca0012_surfCP_a0.dat',skip_header=1,unpack=True)

#Splits the data into lower and upper parts
#Assumes the first sequence of x before repeating is the upper part
SCp1_0u = np.array([SCp1_0[79 - i] for i in range(0,80)]) #Upper pressure coeff
x1_u = [x1[79 - i] for i in range(0,80)]

SCp1_0l = np.array([SCp1_0[i + 80] for i in range(0,80)]) #Lower pressure coeff
x1_l = [x1[i + 80] for i in range(0,80)]



#NACA 0012 AOA 8 deg
#Same process as above
x1, SCp1_8 = np.genfromtxt('data/naca0012_surfCP_a8.dat',skip_header=1,unpack=True)

SCp1_8u = np.array([SCp1_8[79 - i] for i in range(0,80)]) #Upper pressure coeff

SCp1_8l = np.array([SCp1_8[i + 80] for i in range(0,80)]) #Lower pressure coeff



#NACA 0012 dydx and y values
#Evaluates dydx values using above derivative of thickness function. 
#Lower dydx is merely the negative of the Upper dydx
dy1dx_u = [dzt_dx(x) for x in x1_u]
dy1dx_l = [-x for x in dy1dx_u]

#Evaluates y values
y1_u = [zt(x) for x in x1_u]
y1_l = [-x for x in y1_u]

#Evaluates all coeff. using above funcs.
#NACA 0012 0 degrees
Cn1_0 = Cn(SCp1_0l, x1_l, SCp1_0u, x1_u)
Ca1_0 = Ca(SCp1_0l, x1_l, dy1dx_l, SCp1_0u, x1_u, dy1dx_u)
CM1_0 = CM(SCp1_0l, x1_l, dy1dx_l, y1_l, SCp1_0u, x1_u, dy1dx_u, y1_u)
Cl1_0, Cd1_0 = aero_C(Cn1_0, Ca1_0, 0)
#xcp1_0 does not exist. No normal force or moment.
print('Cn1_0: ' + str(Cn1_0))
print('Ca1_0: ' + str(Ca1_0))
print('CM1_0: ' + str(CM1_0))
print('Cl1_0: ' + str(Cl1_0))
print('Cd1_0: ' + str(Cd1_0))
print('')

#NACA 0012 8 degrees
Cn1_8 = Cn(SCp1_8l, x1_l, SCp1_8u, x1_u)
Ca1_8 = Ca(SCp1_8l, x1_l, dy1dx_l, SCp1_8u, x1_u, dy1dx_u)
CM1_8 = CM(SCp1_8l, x1_l, dy1dx_l, y1_l, SCp1_8u, x1_u, dy1dx_u, y1_u)
Cl1_8, Cd1_8 = aero_C(Cn1_8, Ca1_8, 8)
xcp1_8 = xcp(CM1_8,Cn1_8)
print('Cn1_8: ' + str(Cn1_8))
print('Ca1_8: ' + str(Ca1_8))
print('CM1_8: ' + str(CM1_8))
print('Cl1_8: ' + str(Cl1_8))
print('Cd1_8: ' + str(Cd1_8))
print('xcp1_8: ' + str(xcp1_8))
print('')

#NACA 2412 Angle of Attack 0 deg
#Same process as above
x2, SCp2_0 = np.genfromtxt('data/naca2412_surfCP_a0.dat',skip_header=1,unpack=True)

#1st to 81st data for upper, and 82nd to 160th data for lower
SCp2_0u = np.array([SCp2_0[81 - i] for i in range(0,82)]) #Upper pressure coeff
x2_u = [x2[81 - i] for i in range(0,82)]

SCp2_0l = np.array([SCp2_0[i + 82] for i in range(0,78)]) #Lower pressure coeff
x2_l = [x2[i + 82] for i in range(0,78)]



#NACA 2412 Angle of Attack 8 deg
#Same process as above
x2, SCp2_8 = np.genfromtxt('data/naca2412_surfCP_a8.dat',skip_header=1,unpack=True)

SCp2_8u = np.array([SCp2_8[81 - i] for i in range(0,82)]) #Upper pressure coeff

SCp2_8l = np.array([SCp2_8[i + 82] for i in range(0,78)]) #Lower pressure coeff



#NACA 0012 dydx and y values
#Calculates dydx values for NACA 0012
#Because of camber, the derivative of camber function is used
#Depending on upper or lower, derivative of thickness is added or subtracted from camber derivative
dy2dx_u = [dzc_dx(x) + dzt_dx(x) for x in x2_u]
dy2dx_l = [dzc_dx(x) - dzt_dx(x) for x in x2_l]

#Same as above
y2_u = [zc(x) + zt(x) for x in x2_u]
y2_l = [zc(x) - zt(x) for x in x2_l]

#Same process as above
Cn2_0 = Cn(SCp2_0l, x2_l, SCp2_0u, x2_u)
Ca2_0 = Ca(SCp2_0l, x2_l, dy2dx_l, SCp2_0u, x2_u, dy2dx_u)
CM2_0 = CM(SCp2_0l, x2_l, dy2dx_l, y2_l, SCp2_0u, x2_u, dy2dx_u, y2_u)
Cl2_0, Cd2_0 = aero_C(Cn2_0, Ca2_0, 0)
xcp2_0 = xcp(CM2_0,Cn2_0)
print('Cn2_0: ' + str(Cn2_0))
print('Ca2_0: ' + str(Ca2_0))
print('CM2_0: ' + str(CM2_0))
print('Cl2_0: ' + str(Cl2_0))
print('Cd2_0: ' + str(Cd2_0))
print('xcp2_0: ' + str(xcp2_0))
print('')

Cn2_8 = Cn(SCp2_8l, x2_l, SCp2_8u, x2_u)
Ca2_8 = Ca(SCp2_8l, x2_l, dy2dx_l, SCp2_8u, x2_u, dy2dx_u)
CM2_8 = CM(SCp2_8l, x2_l, dy2dx_l, y2_l, SCp2_8u, x2_u, dy2dx_u, y2_u)
Cl2_8, Cd2_8 = aero_C(Cn2_8, Ca2_8, 8)
xcp2_8 = xcp(CM2_8,Cn2_8)
print('Cn2_8: ' + str(Cn2_8))
print('Ca2_8: ' + str(Ca2_8))
print('CM2_8: ' + str(CM2_8))
print('Cl2_8: ' + str(Cl2_8))
print('Cd2_8: ' + str(Cd2_8))
print('xcp2_8: ' + str(xcp2_8))
print('')

#Plots
pl.figure()
s, l = pl.plot(x1_u, -SCp1_0u, 'b.-', x1_l, -SCp1_0l, 'g.-')
pl.title('NACA 0012 Surface Pressure Profile at 0$ \degree $ $ \\alpha $')
pl.xlabel('x/c')
pl.ylabel('-Cp')
pl.legend((s, l), ('Upper Surface','Lower Surface'))

pl.figure()
s, l = pl.plot(x1_u, -1*SCp1_8u, 'b.-', x1_l, -1*SCp1_8l, 'g.-')
v = pl.vlines(xcp1_8,min(-SCp1_8) - 0.3, max(-SCp1_8) + 0.15,linestyles='dashed')
pl.title('NACA 0012 Surface Pressure Profile at 8$ \degree $ $ \\alpha $')
pl.xlabel('x/c')
pl.ylabel('-Cp')
pl.legend((s, l, v), ('Upper Surface','Lower Surface', 'Center of Pressure'))

pl.figure()
s, l = pl.plot(x2_u, -1*SCp2_0u, 'b.-', x2_l, -1*SCp2_0l, 'g.-')
v = pl.vlines(xcp2_0,min(-SCp2_0) - 0.15, max(-SCp2_0) + 0.3,linestyles='dashed')
pl.title('NACA 2412 Surface Pressure Profile at 0$ \degree $ $ \\alpha $')
pl.xlabel('x/c')
pl.ylabel('-Cp')
pl.legend((s, l, v), ('Upper Surface','Lower Surface', 'Center of Pressure'))

pl.figure()
s, l = pl.plot(x2_u, -1*SCp2_8u, 'b.-', x2_l, -1*SCp2_8l, 'g.-')
v = pl.vlines(xcp2_8,min(-SCp2_8) - 0.3, max(-SCp2_8) + 0.3,linestyles='dashed')
pl.title('NACA 2412 Surface Pressure Profile at 8$ \degree $ $ \\alpha $')
pl.xlabel('x/c')
pl.ylabel('-Cp')
pl.legend((s, l, v), ('Upper Surface','Lower Surface', 'Center of Pressure'))

pl.show()
