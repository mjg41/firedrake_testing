from firedrake import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

# Set up mesh
R0 = 1.0
mesh = CubedSphereMesh(radius=R0, refinement_level=4, degree=3)

# Mesh coordinates and cell normals
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

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

# Define function spaces
V = FunctionSpace(mesh, "RTCF", 2)
Q = FunctionSpace(mesh, "DQ", 1)

W = V*Q

# Set up trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Solution function
w = Function(W)

# Initial condition
# Gaussian displacement and zero velocity

u_value = Constant(0.0)
u_expr = as_vector([u_value,u_value,u_value])
p_expr = 10*exp(-100*(x[0]**2 + x[1]**2))
u0 = Function(V).project(u_expr)
p0 = Function(Q).project(p_expr)

outfile = File('output/test.pvd')
# outfile.write(u0)
outfile.write(p0)

# Timestepping
T = 1.0
dt = T/100.0
dtc = Constant(dt)
theta = 0.5
thetac = Constant(theta)

# Store displacement fields for plotting later
us = [u0.copy(deepcopy=True)]
ps = [p0.copy(deepcopy=True)]

# Set up system
swe_eqn = (inner(v,u) - inner(v,u0) - (1 - theta)*c_g*dtc*p*div(v) - theta*c_g*dtc*p0*div(v)
           + p*q - p0*q + (1 - theta)*c_g*dtc*q*div(u) + theta*c_g*dtc*q*div(u0))*dx
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

w.assign(0.0)
t = 0.0
step = 0
output_freq = 5
while t < T:
    swe_solver_gmres.solve()
    u,p = w.split()
    u0.assign(u)
    p0.assign(p)
    
    step += 1
    t += dt
    
    if step % output_freq == 0:
        
        us.append(u.copy(deepcopy=True))
        ps.append(p.copy(deepcopy=True))
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

z_lim = max(abs(z_min), abs(z_max))

ax = fig.add_axes([0,0,1,1], projection='3d')
ax.set_zlim(-z_lim, z_lim)
ax.elev = 30.
ax.azim = -85.

# Setup plot with eta_0

surf = trisurf(ps[0], cmap='bwr', linewidth=0, antialiased=False, vmin=-z_lim, vmax=z_lim, axes=ax)
fig.colorbar(surf)

# Animation function
    
def animate(i, p, plot):
    ax.clear()
    plot = trisurf(p[i], cmap='bwr', linewidth=0, antialiased=False, vmin=-z_lim, vmax=z_lim, axes=ax)
    # ~ ax.set_zlim(z_min, z_max)
    ax.elev = 30.
    ax.azim = -85.
    return plot,

# Call the animator

interval = 1e4*output_freq*dt
anim = FuncAnimation(fig, animate, fargs = (ps, surf), frames=len(ps), interval=interval)

writer=FFMpegWriter(bitrate=5000, fps=6)
anim.save('lin_swe_2d_sphere.mp4', dpi=300, writer = writer)
