from firedrake import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

# Set up mesh - we need hierarchy to use gmg
R0 = 1.0
m = CubedSphereMesh(radius=R0, refinement_level=3, degree=3)
nlevels=2
mh = MeshHierarchy(m, nlevels)
mesh = mh[-1]

# Mesh coordinates and constants
x = SpatialCoordinate(mesh)
global_normal = x
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
                                                    
hybrid_parameters = {'ksp_type': 'preonly',
                        'mat_type': 'matfree',
                        'pc_type': 'python',
                        'pc_python_type': 'firedrake.HybridizationPC',
                        # Solver for the trace system
                        'hybridization': {'ksp_type': 'bcgs',
                                        'pc_type': 'python',
                                        'pc_python_type': 'firedrake.GTMGPC',
                                        'gt': {'mat_type': 'aij',
                                                'pc_mg_log': None,
                                                'mg_levels': {'ksp_type': 'chebyshev',
                                                            #'ksp_richardson_scale':0.6,
                                                            'ksp_max_it': 2,
                                                            'pc_type': 'sor'},
                                                'mg_coarse': {'ksp_type': 'preonly',
                                                              'pc_type': 'mg',
                                                              'pc_mg_cycle_type': 'v',
                                                              'mg_levels': {'ksp_type': 'chebyshev',
                                                                            'ksp_max_it': 2,
                                                                            'pc_type': 'bjacobi',
                                                                            'sub_pc_type': 'sor'}}}}}

# Define P1 coarse space and callback

def get_coarse_space():
    return FunctionSpace(mesh, 'CG', 1)

def coarse_callback():
    P1 = get_coarse_space()
    q = TrialFunction(P1)
    r = TestFunction(P1)
    beta = dt*0.5

    return (inner(q, r) +
            beta**2*c_g**2*
            inner(grad(q), grad(r)))*dx

# V_trace = FunctionSpace(mesh, "HDiv Trace", args.degree)
# interpolation_matrix = prolongation_matrix(V_trace,get_coarse_space())

appctx = {'get_coarse_operator': coarse_callback,
          'get_coarse_space': get_coarse_space}# ,
            # 'interpolation_matrix':interpolation_matrix}

# hybrid_parameters = {'ksp_type': 'preonly',
#                      'mat_type': 'matfree',
#                      'pc_type': 'python',
#                      'pc_python_type': 'firedrake.HybridizationPC',
#                      # Solver for the trace system
#                      'hybridization': {'ksp_type': 'preonly', 'pc_type': 'lu'}}

swe_solver_gmres = LinearVariationalSolver(swe_problem,
                                           solver_parameters=hybrid_parameters,
                                           appctx=appctx)

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

ax = fig.add_axes([0,0.025,1,0.95], projection='3d')
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
anim.save('lin_swe_2d_hyb_sphere.mp4', dpi=300, writer = writer)
