#Imports
import matplotlib.pyplot as pl
import math as m
import numpy as np

#Problem 1

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

steps = 1000 #Amount of x values to generate
xcr = range(0, 1*steps + 1) #Generate x vector using range
xcr = [x/steps for x in xcr] #Normalize the x vector


zt1u = [zt(x) for x in xcr] #Upper thickness for NACAXX12. Pass each xcr value to defined function
zt1b = [-x for x in zt1u] #Lower thickness
zc1 = [x*0 for x in xcr] #Chord position for NACA 0012 Symmetric, generates 0s with same length as xcr


zc2 = [zc(x) for x in xcr] #Chord position for NACA 2412 asymmetric. Pass each xcr value to defined function
dzc2_dx = [dzc_dx(x) for x in xcr]
tri = [m.cos(m.atan(x)) for x in dzc2_dx]
print('Strongest camber slope effect on z_u/l: ' + str(min(tri)))
zt2u = [zc2[x] + zt1u[x] for x in range(0,len(zc2))] #Calculate upper airfoil profile distance
zt2b = [zc2[x] - zt1u[x] for x in range(0,len(zc2))] #Calculate lower airfoil profile distance

#Plots
pl.figure()
pl.axis('equal')
pl.plot(xcr, zt1u, 'b', xcr, zt1b, 'b')
pl.plot(xcr, zc1, 'b--')
pl.title('NACA 0012 Airfoil Profile')
pl.xlabel('x')
pl.ylabel('z')

pl.figure()
pl.axis('equal')
pl.plot(xcr, zt2u, 'b', xcr, zt2b, 'b')
pl.plot(xcr,zc2, 'b--')
pl.title('NACA 2412 Airfoil Profile')
pl.xlabel('x')
pl.ylabel('z')

pl.show()

