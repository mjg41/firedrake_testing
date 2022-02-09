from firedrake import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# Set up mesh
nx = 100
ny = nx
mesh = UnitSquareMesh(nx,ny, quadrilateral=True)

# Mesh coordinates and constants
x,y = SpatialCoordinate(mesh)
dphi = 0.1
r_p = 0.45
r_m = 0.05
sigma = 0.25
# Reference length: radius of Earth = 6.370E6 [m]
Rearth = 6.370E6
Xref = Rearth
# Gravitational acceleration g_grav = 9.81 [ms^{-2}]
g_grav = 9.81
# Depth of fluid = 2.0E3 [m]
H = 2.0E3
# Reference time = 1 day = 24*3600 [s]
day = 24*3600
Tref = day
c_g = Tref / Xref * np.sqrt(g_grav*H)
# angular frequency of Earth rotation
f_coriolis = 4.0*np.pi/day*Tref
f_over_cg = f_coriolis/c_g

print(f_over_cg)

# Define function spaces
V = FunctionSpace(mesh, "RTCF", 2)
Q = FunctionSpace(mesh, "DQ", 1)

W = V*Q

# Set up trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Solution function
w = Function(W)

# Initial conditions

# Bathymetry

def bath_plot(r):

    if ((r <= r_m) or (r_p <= r)):
        height = 1.0
    else:
        exponent =  1.0/(r - r_p)
        exponent += 4.0/(r_p - r_m)
        exponent -= 1.0/(r - r_m)
        height = 1.0 + dphi*np.exp(exponent)
    return height
    
# Initial displacement field - this should be preserved

def phi_plot(r):

    if (r <= r_m):
        height = 1.0-dphi
    elif (r >= r_p):
        height = 1.0
    else:
        # Hyperbolic angle
        hangle = sigma/(r - r_m)
        hangle += sigma/(r - r_p)
        slope = dphi*(1.0 + np.tanh(hangle))
        height = 1.0 - 0.5*slope
    return height
    
# Initial velocity field - this should be preserved

def u_plot(r):
    
    if ( (r <= r_m) or (r_p <= r)):
        velocity = 0.0
    else:
        dphi_dr = 0.5*sigma*dphi
        dphi_dr *= (1.0/(r - r_m)**2 + 1.0/(r - r_p)**2)
        dphi_dr *= (1.0 - np.tanh(sigma/(r - r_m) + sigma/(r - r_p))**2)
        
        velocity = 1.0/f_over_cg*(phi(r) + bath(r))*dphi_dr
    return velocity

# Bathymetry in UFL

def bath_ufl(mesh):

    x,y = SpatialCoordinate(mesh)
    x -= 0.5
    y -= 0.5
    r = sqrt(x**2+y**2)
    
    exponent =  1/(r - r_p)
    exponent += 4/(r_p - r_m)
    exponent -= 1/(r - r_m)
    bump = 1.0 - dphi*exp(exponent)
    height = conditional(r<=r_m, 1.0,
                   conditional(r>=r_p, 1.0,
                               bump))
    return height

# Displacement in UFL

def phi_ufl(mesh):

    x,y = SpatialCoordinate(mesh)
    x -= 0.5
    y -= 0.5
    r = sqrt(x**2+y**2)
    
    return conditional(r<=r_m,-dphi,
                       conditional(r>=r_p,0.0,
                                   -0.5*dphi*(1.0+tanh(sigma/(r-r_m)+sigma/(r-r_p)))))

# Velocity in UFL

def u_ufl(mesh):

    x,y = SpatialCoordinate(mesh)
    x -= 0.5
    y -= 0.5
    r = sqrt(x**2+y**2)
    
    potential = phi_ufl(mesh) + bath_ufl(mesh)
    dphi_dr = 0.5*sigma*dphi*(1.0/(r-r_m)**2+1.0/(r-r_p)**2)
    dphi_dr*= (1.0-tanh(sigma/(r-r_m)+sigma/(r-r_p))**2)
    u_r_expr = (1.0/f_over_cg)*potential*(dphi_dr/r)
    return conditional(Or(r<=r_m,r_p<=r),as_vector((0.0,0.0)),
                       as_vector((-y*u_r_expr,x*u_r_expr)))
                       
# Perp function

def perp(u):
    
    return as_matrix([[0,-1],[1,0]])*u

# Initialise

u_expr = u_ufl(mesh)
p_expr = phi_ufl(mesh)
u0 = Function(V).project(u_expr)
p0 = Function(Q).project(p_expr)

# Assign w to ICs

w.sub(0).assign(u0)
w.sub(1).assign(p0)

# Timestepping
T = 1.0
dt = T/100.0
dtc = Constant(dt)
theta = 0.5
thetac = Constant(theta)
t = 0.0
step = 0
output_freq = 5

# Store displacement fields for plotting later
# Store velocities for debugging
ps = [p0.copy(deepcopy=True)]
us = [u0.copy(deepcopy=True)]

# Output file
# ~ outfile = File('output/lin_swe_2d_bath.pvd')
# ~ u_fn, p_fn = w.split()
# ~ u_fn.rename('momentum')
# ~ p_fn.rename('potential')
# ~ outfile.write(u_fn, p_fn, time=t)

# Set up system
swe_eqn = (inner(v,u) - inner(v,u0)
               - (1 - thetac)*dtc*(c_g*bath_ufl(mesh)*p*div(v) - inner(v, f_coriolis*perp(u)))
               - thetac*dtc*(c_g*bath_ufl(mesh)*p0*div(v) - inner(v, f_coriolis*perp(u0)))
           + p*q - p0*q
               + (1 - thetac)*c_g*dtc*q*div(u)
               + thetac*c_g*dtc*q*div(u0))*dx

a = lhs(swe_eqn)
L = rhs(swe_eqn)

swe_problem = LinearVariationalProblem(a, L, w, constant_jacobian=False)

solver_parameters = {'ksp_type': 'gmres',
                     'ksp_rtol': 1.0e-7,
                     'ksp_max_it': 1500,
                     'pc_type': 'fieldsplit',
                     'pc_fieldsplit': {'type': 'schur',
                                       'schur_fact_type': 'full',
                                       'schur_precondition': 'selfp'},
                     'fieldsplit_0': {'ksp_type': 'preonly',
                                      'pc_type': 'bjacobi',
                                      'sub_pc_type': 'ilu'},
                     'fieldsplit_1': {'ksp_type': 'preonly',
                                      'pc_type': 'gamg',
                                      'mg_levels': {'ksp_type': 'chebyshev',
                                                    'ksp_max_it': 5,
                                                    'pc_type': 'bjacobi',
                                                    'sub_pc_type': 'ilu'}}}
swe_solver_gmres = LinearVariationalSolver(swe_problem,
                                           solver_parameters=solver_parameters)

# Solve problem

while t < T:
    swe_solver_gmres.solve()
    u,p = w.split()
    u0.assign(u)
    p0.assign(p)
    
    step += 1
    t += dt
    
    if step % output_freq == 0:
        
        ps.append(p.copy(deepcopy=True))
        us.append(u.copy(deepcopy=True))
        u_fn, p_fn = w.split()
        # ~ outfile.write(u_fn, p_fn, time=t)
        print('t = {:.4f}'.format(t))

## Animation ##

# Setup phase

fig = plt.figure()

z_min = 0.0
z_max = 0.0
for p in ps:
    new_max = max(p.vector())
    new_min = min(p.vector())
    if new_max > z_max:
        z_max = new_max
    if new_min < z_min:
        z_min = new_min

ax = fig.gca(zlim=(z_min, z_max), projection='3d')
ax.elev = 30.
ax.azim = -85.

# Setup plot with eta_0

surf = trisurf(ps[0], linewidth=0, antialiased=False, axes=ax)
fig.colorbar(surf)

# Animation function
    
def animate(i, p, plot):
    ax.clear()
    plot = trisurf(p[i], linewidth=0, antialiased=False, axes=ax)
    ax.set_zlim(z_min, z_max)
    ax.elev = 30.
    ax.azim = -85.
    return plot,

# Call the animator

interval = 1e4*output_freq*dt
anim = FuncAnimation(fig, animate, fargs = (ps, surf), frames=len(ps), interval=interval)

writer=FFMpegWriter(bitrate=5000, fps=6)
anim.save('lin_swe_2d_bath.mp4', dpi=300, writer = writer)
