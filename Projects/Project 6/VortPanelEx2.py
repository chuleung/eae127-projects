"""VORTEX PANEL METHOD
Logan Halstrom
EAE 127
CREATED: 22 NOV 2014
MODIFIY: 05 NOV 2015


Description:  Simulate geometry in lifting potential flow with constant
strength vortex panels.  Satisfy kutta condition as additional constraint on
panel linear system of equations.  Avoid over-constrained system by ignoring
tangency condition for one "missing" panel.
    Solve for vortex strength distributions of one airfoil.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

###############################################################################
### DATA CLASSES ##############################################################
###############################################################################

class Airfoil:
    def __init__(self, xgeom, ygeom, ihole, alpha_deg, Vinf, rho_inf, name='Airfoil'):
        """Constants pertaining to airfoil and freestream conditions
        Vinf --> freestream velocity (default = 1)
        alpha_deg --> angle of attack in degrees (default 0)
        xgeom, ygeom --> coordinates of airfoil geometry
        name --> string of airfoil name, used for titles
        """
        #airfoil geometry coordinates
        self.xgeom, self.ygeom = xgeom, ygeom
        #chord
        self.c = (xgeom.max() - xgeom.min())
        #index of panel to leave out of tangency condition
        self.ihole = ihole

        #AoA [deg]
        self.alpha_deg = alpha_deg
        #AoA in radians
        self.alpha = alpha_deg*np.pi/180
        #freestream velocity
        self.Vinf = Vinf
        #freestream density
        self.rho = rho_inf

        #airfoil name
        self.name = name
        self.title = ''
        #run details for savefile name
        self.details = ''

class Panel:
    def __init__(self, xa, ya, xb, yb):
        """Initialize panel
        (xa, ya) --> first end-point of panel
        (xb, yb) --> second end-point of panel
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        #CONTROL-POINT (center-point)
        self.xc, self.yc = (xa + xb) / 2, ( ya + yb ) / 2
        #LENGTH OF THE PANEL
        self.length = ((xb - xa)**2 + (yb - ya)**2) ** 0.5

        #INITIALIZE:
        #VORTEX STRENGTH DISTRIBUTION
        self.gamma = 0.
        #TANGENTIAL VELOCITY AT PANEL SURFACE (CONTROL POINT)
        self.vt = 0.
        #PRESSURE COEFFICIENT AT PANEL SURFACE (CONTROL POINT)
        self.Cp = 0.

        #ORIENTATION OF THE PANEL (angle between x-axis and panel's normal)
        if xb-xa <= 0.:
            self.beta = np.arccos((yb-ya)/self.length)
        elif xb-xa > 0.:
            self.beta = np.pi + np.arccos(-(yb-ya)/self.length)

        #PANEL ON UPPER OR LOWER SURFACE (for plotting surface pressure)
        if self.beta <= np.pi:
            self.surf = 'upper'
        else:
            self.surf = 'lower'

###############################################################################
### PANEL DISCRETIZATION FUNCTIONS ############################################
###############################################################################

def MakePanels(xgeom, ygeom, n_panel, method):
    """Discretize geometry into panels using various methods.
    Return array of panels.
    n_panel --> number of panels
    method --> panel discretization method:
                'constant' --> interpolate points with constant spacing in x
                'circle' --> map circle points to airfoil,
                'given' --> use given distribution
    """

    def ConstantSpacing(xgeom, ygeom, n_panel):
        """Creates airfoil panel points with uniform x-spacing.
        (Finer spacing at TE to enforce kutta).  Requires odd number of panels.
        x, y --> geometry coordinates
        n_panel --> number of panels to distretize into
        """
        if n_panel%2 == 0:
            print('WARNING: Constant spacing method requires odd number of panels')
        c = xgeom.max() - xgeom.min()
        #Ratio of TE panel length to uniform spacing length
        frac = 0.25
        #get TE panel lengths
        TE_length = frac * c / ( ( (n_panel + 1) / 2 ) - 2 + frac )

        #MAKE UNIFORM SPACING VECTOR FOR NON-TE PANELS
        #Start at some non-zero x for panel normal to flow in front at LE
        xLE = xgeom.min() + 0.001 * c
        #End is where TE panels start
        xTE = xgeom.max() - TE_length
        #create x points
            #(number of points is half (one surface) of amount needed to end up
            #with n_panels after accounting for TE panels
        xnew = np.linspace(xLE, xTE, (n_panel+1)/2-1)
        #ADD TE POINT (distance from xnew[-2] to xnew[-1] is TE_length)
        xnew = np.append(xnew, xgeom.max())

        #INTERPOLATE GEOMETRY FOR NEW X-SPACING
        xoldup, yoldup, xoldlo, yoldlo = GeomSplit(xgeom, ygeom)
        #linear interpolate new y points
        ynewup = np.interp(xnew, xoldup, yoldup)
        ynewlo = np.interp(xnew, xoldlo, yoldlo)
        #Merge new coordinates into one set
        yends = GeomMerge(xnew, ynewup, xnew, ynewlo)[1]
        xends = np.empty_like(yends)
        xends[0:len(xnew)] = xnew[-1::-1]
        xends[len(xnew):] = xnew[:]
        # yends[n_panel] = yends[0]
        return xends, yends

    def CircleMethod(xgeom, ygeom, n_panel):
        """Create x-spacing based on a circle, then map y-points onto body
        geometry using linear interpolation.  Some panel spacings cause
        interpolation to break, so check the resulting geometry for each panel#
        x, y --> geometry coordinates
        n_panel --> number of panels to distretize into
        """

        #Make circle with diameter equal to chord of airfoil
        R = (xgeom.max() - xgeom.min()) / 2
        center = (xgeom.max() + xgeom.min()) / 2
        #x-coordinates of circle (also new x-coordinates of airfoil)
            #(n+1 because first and last points are the same)
        xends = center + R*np.cos(np.linspace(0, 2*np.pi, n_panel+1))
        #for each new x, find pair of original geometry points surrounding it
        #and linear interpolate to get new y

        #append first point of original geometry to end so that i+1
        #at last point works out
        xgeom, ygeom = np.append(xgeom, xgeom[0]), np.append(ygeom, ygeom[0])

        #Get index of split in circle x
        i = 0
        while i < len(xends):
            if (xends[i] - xends[i-1] < 0) and (xends[i+1] - xends[i] > 0):
                isplit = i
                break
            i += 1
        #Split upper and lower surfaces for interpolation
        xnewup = xends[isplit::-1]
        xnewlo = xends[isplit+1:]
        xoldup, yoldup, xoldlo, yoldlo = GeomSplit(xgeom, ygeom)
        #linear interpolate new y points
        ynewup = np.interp(xnewup, xoldup, yoldup)
        ynewlo = np.interp(xnewlo, xoldlo, yoldlo)
        #Merge new coordinates into one set
        yends = GeomMerge(xnewup, ynewup, xnewlo, ynewlo)[1]
        # yends[n_panel] = yends[0]

        #***************************might need to make trailing edge one point*
        return xends, yends

    if method == 'constant':
        #Constant spacing method Method
        xends, yends = ConstantSpacing(xgeom, ygeom, n_panel)
    elif method == 'circle':
        #Circle Method
        xends, yends = CircleMethod(xgeom, ygeom, n_panel)
    elif method == 'given':
        #use points as given
        xends, yends = xgeom, ygeom
        n_panel = len(xends) - 1

    print('Upper/Lower y-coordinate at TE: (', xends[0],',', yends[0], ') ; (',
                                               xends[0],',', yends[-1], ')')
    #assign panels
    panels = np.zeros(n_panel, dtype=object)
    for i in range(0, n_panel):
        panels[i] = Panel(xends[i], yends[i], xends[i+1], yends[i+1])

    return panels

def VortexPanelNormalIntegral(pi, pj):
    """Evaluate contribution of panel j at control point of panel i in
    panel i's normal direction.
    pi --> panel which is being contributed to
    pj --> panel which is contributing to other panel's normal velocity
    """
    #function to integrate
    def func(s):
        #s is the line of points from a panel start point to end point.
        #We need to integrate along s:

        #x-coordinates of s-vector along panel
        xjsj = pj.xa - np.sin(pj.beta)*s
        #y-coordinates of s-vector along panel
        yjsj = pj.ya + np.cos(pj.beta)*s
        #We need to calculate contribution of each x,y along panel j.
        #that's what xjsj yjsj are: the points along panel j.

        #The partial derivative of our integral with respect to the normal
        #direction produces these partial derivative terms, which can be
        #calculated using simple trigonometry from the angle between the
        #unit vectors.
        dydn = np.sin(pi.beta)
        dxdn = np.cos(pi.beta)

        #Now, let's make a vector that is the equation for the normal velocity
        #produced by each x,y pair along panel j and panel i's control point
        return (-1/(2 * np.pi) * ( (pi.xc - xjsj)*dydn - (pi.yc - yjsj)*dxdn ) / ( (pi.xc - xjsj) ** 2 + (pi.yc - yjsj) ** 2) )
        #pi.xc, pi.yc are the control points of panel i
        #xjsj, yjsj are the points along panel j
        #gamma isnt in this equation because we will multiply it in later with
            #our linear system of equations

    #Integrate along length of panel
    return integrate.quad(lambda s:func(s), 0., pj.length)[0]

###############################################################################
### SOLVE PANEL SYSTEM ########################################################
###############################################################################

def SolveVorticity(panels, airfoil):
    """Solve for the source strength distributions and vortex sheet such that
    the tangency condition and the Kutta condition are satified.
    """

    def KuttaCondition(panels):
        """
        KUTTA CONDITION:
            Tangential velocities of upper TE (Panel 1) and lower TE (Panel N) are
        the same assuming bernoulli and same pressure because shared point (TE).
            Vorticity of a vortex sheet: gamma = utop - ubot (Anderson eqn 4.8).
        Since utop = ubot for TE, ------> gammaTE = 0 <--------- that's the K.C.
        """

        n_panel = len(panels)
        #To make vorticity of TE top and TE bottom panel add to zero,
        #row entry will be array of zeros with first and last entry as 1's,
        #thus only accounting for TE top and bottom.  set equal to zero for KC
        x = np.zeros(n_panel)
        x[0], x[-1] = 1, 1
        return x

    def TangencyCondition(panels, airfoil):
        """Array accounting for vortex contribution from each panel.
        panels --> array of panels
        """
        n_panel = len(panels)
        #Populate matrix of system of equations for each panel
        size = (n_panel, n_panel)
        A = np.zeros(size)
        #Fill diagonal with term for each panel's contribution to itself
        np.fill_diagonal(A, 0.)
        #Fill matrix with each panel's contribution to every other panel
        for i, pi in enumerate(panels):
            for j, pj in enumerate(panels):
                if i != j:
                    A[i,j] = VortexPanelNormalIntegral(pi, pj)

        #Replace Missing Panel with Kutta Condition
        A[airfoil.ihole,:] = KuttaCondition(panels)

        return A

    #Make system to solve

    """
          |   vortex contributions    |  vorticies  | freestream normal |
          |                           |             |    contribution   |

           j=1 j=2 . .  . . j=N-1 j=N
    i=1   [ 0                        ]   [ gamma1  ]   [ -Vinf,n1 ]
          [                vortex    ]   [         ]   [ -Vinf,n2 ]
    i=2   [     0         contrib    ]   [ gamma2  ]   [     .    ]
     .    [       .                  ]   [    .    ]   [     .    ]
     .    [                          ]   [    .    ]   [          ]
    i=ik  [ 1   0  . .  . .  0   1   ] * [ gammaik ] = [     0    ]<--K.C.
     .    [                          ]   [    .    ]   [          ]
     .    [                .         ]   [    .    ]   [          ]
    i=N-1 [   vortex          0      ]   [ gammaN-1]   [     .    ]
          [   contrib                ]   [         ]   [     .    ]
    i=N   [                        0 ]   [ gammaN  ]   [ -Vinf,nN ]


                    [A]           *       [gam]      =        [b]

            NOTE:  ik is index of missing panel to enforce kutta condition
    """

    n_panel = len(panels)
    #CONTRIBUTION OF EACH PANEL TO FLOW TANGENCY OF EACH CONTROL POINT
    A = TangencyCondition(panels, airfoil)
    #Right Hand Side
    b = np.empty(n_panel, dtype=float)
    for i, p in enumerate(panels):
        #FREESTREAM FLOW IN NORMAL DIR, MUST BE CANCELED BY VORTEX CONTRIBUTIONS
        b[i] = -airfoil.Vinf * np.cos(airfoil.alpha - p.beta)
    #KUTTA CONDITION (gamTE = 0)
    b[airfoil.ihole] = 0

    #SOLVE SYSTEM
    gam = np.linalg.solve(A, b)

    #assign variables
    for i, p in enumerate(panels):
        p.gamma = gam[i]

    #Check for Kutta Condition
    print('gamma1 =', panels[0].gamma,
          'gammaN =', panels[-1].gamma,
          'gamma1 + gammaN =', panels[0].gamma + panels[-1].gamma)

    return panels

def TangencyCheck(panels, airfoil, show_plot):
    """Calculate velocity normal to body surface, should be zero.

    panels --> array of panels
    method --> 'integrate' (integrate contributions from all panels),
                'gamma' (panel surface velocity equals panel vorticity)
    """
    #dont plot if show_plot=0
    if show_plot==0:
        return

    n_panel = len(panels)

    #Populate matrix of system of equations for each panel
    size = (n_panel, n_panel)
    A = np.zeros(size)
    #Fill diagonal with term for each panel's contribution to itself
    np.fill_diagonal(A, 0.)
    #Fill matrix with each panel's contribution to every other panel
    for i, pi in enumerate(panels):
        for j, pj in enumerate(panels):
            if i != j:
                #tangential vel. contrib. at all points from all vortex dists.
                A[i,j] = VortexPanelNormalIntegral(pi, pj)

    #Populate b vector
    b = airfoil.Vinf * np.cos([airfoil.alpha - p.beta for p in panels])
    #vector of vorticity distribution (all panel gammas are the same)
    gam = [p.gamma for p in panels]
    #solve system for normal velocities
    vn = np.dot(A, gam) + b

    #Plot Surface Normal Velocity
    size = 8
    plt.figure(figsize = (size,size))
    # plt.xlim(0, 360)
    plt.title('Normal Velocity Over ' + airfoil.title, fontsize=txt_ttl)
    plt.xlabel(r'$\frac{x}{c}$', fontsize=txt_lbl*1.5)
    plt.ylabel('$V_N$', fontsize=txt_lbl)
    plt.plot([p.xc for p in panels], vn)
    plt.legend(loc='best')
    plt.show()

def PlotPanels(xgeom, ygeom, panels, name, show_plot):
    #dont plot if show_plot=0
    if show_plot==0:
        return

    edgespace = 0.1 * (xgeom.max()-xgeom.min())
    xmin, xmax = xgeom.min()-edgespace, xgeom.max()+edgespace
    ymin, ymax = ygeom.min()-edgespace, ygeom.max()+edgespace
    factor = 12
    size = (factor*(xmax-xmin), factor*(ymax-ymin))
    plt.figure(figsize = size)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(str(len(panels)) + ' Panel Discretization of ' + name,
                                                            fontsize=txt_ttl)
    plt.xlabel(r'$\frac{x}{c}$', fontsize=txt_lbl)
    plt.ylabel(r'$\frac{y}{c}$', fontsize=txt_lbl)
    # plt.grid(True)
    plt.plot(xgeom, ygeom, '--b', label = 'Original Geometry', linewidth=2)
    plt.plot(np.append([p.xa for p in panels], panels[-1].xb),
                        np.append([p.ya for p in panels], panels[-1].yb),
                            'g', label = 'Panel', linewidth=2)
    plt.plot([p.xa for p in panels], [p.ya for p in panels],
                    'go', label = 'End Points', linewidth=1, markersize = mkr)
    plt.plot([p.xc for p in panels], [p.yc for p in panels],
                 'k^', label = 'Center Points', linewidth=1, markersize = mkr)
    plt.legend(loc='best')
    # plt.axis('equal')
    plt.show()

###############################################################################
### GEOMETRY MODIFICATION FUNCTIONS ###########################################
###############################################################################

def ReadXfoilGeometry(ifile):
    """Reads two coulmn xfoil output files. Either geometry or cp distributions
    ifile --> path of input file (string)
    """
    xgeom, ygeom = np.loadtxt(ifile, skiprows=1, unpack=True)
    return xgeom, ygeom

def GeomSplit(x,y):
    """Given mses type data, find location of leading edge and split
    x,y coordintes into two sets (upper and lower)
    """
    #Get index of leading edge on upper surface
    i = 0
    while i < len(x):
        if (np.arctan2(y[i], x[i]) >= 0) and (np.arctan2(y[i+1], x[i+1]) < 0):
            iLE = i
            break
        i += 1
    #Split upper and lower surfaces for interpolation
    xup, yup= x[iLE::-1], y[iLE::-1]
    xlo, ylo= x[iLE+1:], y[iLE+1:]

    return xup, yup, xlo, ylo

def GeomMerge(xup, yup, xlo, ylo):
    """merge upper and lower surface geometry sets into one set of x,y
    (XFOIL format)
    """
    n1 = len(xup)
    n = n1 + len(xlo)
    x, y = np.zeros(n), np.zeros(n)
    #reverse direction of upper surface coordinates
    x[:n1], y[:n1] = xup[-1::-1], yup[-1::-1]
    #append lower surface coordinates as they are
    x[n1:], y[n1:] = xlo, ylo
    return x, y

###############################################################################
### MAIN FUNCTION #############################################################
###############################################################################

#Plotting Constants
txt_lbl = 14                        #label fontsize
txt_ttl = 14                        #title fontsize
mkr = 8

"""
Run simulation for various amounts of panels
geometry --> what geometry to run
name --> string of geometry name
Vinf --> freestream velocity
alpha_deg --> angle of attack
n --> numbers of panels to discretize geometry into
"""

#INPUTS
geometry = 'naca0012.dat'
name = 'NACA0012'
# geometry = 'Data/naca2412.dat'
# name = 'NACA 2412'
# geometry = 'Data/s1223.dat'
# name = 'Selig1223'
Vinf = 1
#freestream density
rho_inf = 1.
alpha_deg = 10
#number of panels
n = 61
#panel to enforce kutta condition at
missingpanel = 6

#READ AIRFOIL DATA
xgeom, ygeom = ReadXfoilGeometry(geometry)

#INITIALIZE FREESTREAM CONDITIONS AND AIRFOIL PARAMETERS
#index of missing panel is one less than number because of zero indexing
ihole = missingpanel - 1
airfoil = Airfoil(xgeom, ygeom, ihole, alpha_deg, Vinf, rho_inf, name)

#DISCRETIZE GEOMETRY INTO PANELS
panels = MakePanels(airfoil.xgeom, airfoil.ygeom, n, 'constant')

#FIGURE TITLES
airfoil.title = (airfoil.name + ' at $\\alpha$='
                                + str(airfoil.alpha_deg) + '$^o$'
                                + ' (' + str(len(panels)) + ' panels, #'
                                + str(airfoil.ihole+1) + ' missing)')
airfoil.details = ('_a' + str(airfoil.alpha_deg) + '_'
                        + str(len(panels)) + 'panels_'
                        + str(airfoil.ihole+1) + 'hole.png')

#PLOT PANEL GEOMETRY
PlotPanels(airfoil.xgeom, airfoil.ygeom, panels, airfoil.name, 0)

#SOLVE FOR PANEL STRENGTH DISTRIBUTIONS
panels = SolveVorticity(panels, airfoil)
#check for flow tangent to body
TangencyCheck(panels, airfoil, 0)

#FROM HERE, USE SOLVED PANEL STRENGTH DISTRIBUTIONS TO CALCULATE VELOCITIES
#ON THE AIRFOIL SURFACE AND IN THE SURROUNDING FLOW FIELD, WHICH CAN BE USED
#IN TURN TO CALCULATE AERODYNAMIC FORCES.

###############################
### FUNCTIONS              ####
###############################

def PanelIntegral2(pj, x, y, dir):
    """Evaluate contribution of panel at a point (x,y).
    pj --> panel from which contribution is calculated
    x --> x-location of where contribution is calculated
    y --> y-location of where contribution is calculated
    direction of component and type of contribution --> 'x source',
                                                        'y vortex',
                                                        etc
    """
    if dir=='x source':
        #derivatives with respect to: x-axis for sources
        dxd_, dyd_ = 1, 0
        coeff = 1
    elif dir=='y source':
        #derivatives with respect to: y-axis for sources
        dxd_, dyd_ = 0, 1
        coeff = 1
    elif dir=='x vortex':
        #derivatives with respect to: x-axis for vortices
        dxd_, dyd_ = 0, 1
        coeff = -1
    elif dir =='y vortex':
        #derivatives with respect to: y-axis for vortices
        dxd_, dyd_ = 1, 0
        coeff = -1
    #function to integrate
    def func(s):
        #x-coord of s-vector along panel
        xjsj = pj.xa - np.sin(pj.beta)*s
        #y-coord of s-vector along panel
        yjsj = pj.ya + np.cos(pj.beta)*s
        return (coeff*0.5/np.pi)*((x - xjsj)*dxd_ + coeff*(y - yjsj)*dyd_) / ((x - xjsj) ** 2 + (y - yjsj) ** 2)
    
    #Integrate along length of panel
    return integrate.quad(lambda s:func(s), 0., pj.length)[0]


def Panel_Point_Vel(panels, x, y):
    """Evaluates the x-,y- velocity components generated by a set of panels at a set of points.
    
    Arguments
    ---------
    panels -- panel from which the contribution is calculated.
    x -- x-locations
    y -- y-locations
    
    Returns
    -------
    u -- x-component of the velocity
    v -- y-component of the velocity
    """
    
    n = len(x)
    if n != len(y):
        print('Must have equal number of x and y values!')
        return
    
    u = np.zeros_like(x,dtype=float)
    v = u
    
    for i in range(0,n):
        #Calculates the velocity at a point
        u[i] = sum([p.gamma*PanelIntegral2(p, x[i], y[i], 'x vortex') for p in panels])
        v[i] = sum([p.gamma*PanelIntegral2(p, x[i], y[i], 'y vortex') for p in panels])
    return u, v


def Gamma_General(panels, airfoil, x, y):
    """Calculates the circulation about a curve when given discretized points on the curve.
        Uses vector components of V and ds. Direction of integration determined by points
    Arguments
    ---------
    panels -- Set of panels generated from airfoil
    airfoil -- An Airfoil class object. Only angle of attack value is used
    x -- x-points of curve
    y -- y-points of curve
    
    Returns
    -------
    Gamma -- Circulation
    """    
    n = len(x) #Number of discretized points for integral
    dx = np.array([0.5*(x[i+1] - x[i-1]) for i in range(1,n-1)],dtype=float)
    dy = np.array([0.5*(y[i+1] - y[i-1]) for i in range(1,n-1)],dtype=float)
    
    dx = np.append(dx,x[n-1]-x[n-2])
    dx = np.insert(dx,0,x[1]-x[0])
    
    dy = np.append(dy,y[n-1]-y[n-2])
    dy = np.insert(dy,0,y[1]-y[0])
    
    up,vp = Panel_Point_Vel(panels,x,y)
    
    Vx = up + airfoil.Vinf*np.cos(airfoil.alpha)
    Vy = vp + airfoil.Vinf*np.sin(airfoil.alpha)
    
    Gamma = np.trapz(Vx,x) + np.trapz(Vy,y) #Evaluates integral
    
    return Gamma

def Gamma_General2(panels, airfoil, x, y):
    """Calculates the circulation about a curve when given discretized points on the curve.
       Uses definition of a dot-product in 2D. Direction of integration determined by points
    Arguments
    ---------
    panels -- Set of panels generated from airfoil
    airfoil -- An Airfoil class object. Only angle of attack value is used
    x -- x-points of curve
    y -- y-points of curve
    
    Returns
    -------
    Gamma -- Circulation
    """    
    n = len(x)-1 #Number of discretized points for integral
    dx = np.array([x[i+1] - x[i] for i in range(0,n)],dtype=float)
    dy = np.array([y[i+1] - y[i] for i in range(0,n)],dtype=float)
    ds = (dx**2 + dy**2)**0.5

    up,vp = Panel_Point_Vel(panels,x[0:n],y[0:n])
    
    Vx = up + airfoil.Vinf*np.cos(airfoil.alpha)
    Vy = vp + airfoil.Vinf*np.sin(airfoil.alpha)
    
    V = (Vx**2 + Vy**2)**0.5

    thi1 = np.arctan2(dy,dx)
    thi2 = np.arctan2(Vy,Vx)
    
    print(len(ds))
    print(len(V))
    print(len(thi1))
    print(len(thi2))

    #Gamma = np.trapz(V*ds*np.cos(thi2-thi1))
    Gamma = np.trapz(V*np.cos(thi2-thi1),dx=ds[0]) #Evaluates integral
    
    return Gamma

#########################################################################
###                    ####
### Script to run with ####
###                    ####
###########################

n_points = 1000
theta = np.linspace(2*np.pi,0, n_points)
R = 5
xc = 0.5
yc = 0.0

x = R*np.cos(theta) + xc #plus translation to center circle
y = R*np.sin(theta) + yc

gamma1 = Gamma_General2(panels,airfoil,x,y)
cl_gen = 2*gamma1/airfoil.Vinf

print(cl_gen)