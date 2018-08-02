from thetis_adjoint import *
from firedrake.petsc import PETSc

from utils.conversion import from_latlon
from utils.options import TohokuOptions
from utils.setup import problem_domain
from utils.sw_solvers import tsunami

# Set up problem
op = TohokuOptions(approach='DWR')
mesh, u0, eta0, b, BCs, f, nu = problem_domain(8, op=op)

# Option 1: a delicate and potentially hazardous piece of infrastructure
op.loc = from_latlon(op.lat("Fukushima Daiichi"), op.lon("Fukushima Daiichi"))
op.radii = [50e3]

# # Option 2: dense population centres
# op.loc = from_latlon(op.lat("Tokyo"), op.lon("Tokyo"))
# op.radii = [150e3]

# Option 3: a collection of important regions
# op.loc = from_latlon(op.lat("Tokai"), op.lon("Tokai")) \
#          + from_latlon(op.lat("Fukushima Daiichi"), op.lon("Fukushima Daiichi")) \
#          + from_latlon(op.lat("Fukushima Daini"), op.lon("Fukushima Daini")) \
#          + from_latlon(op.lat("Onagawa"), op.lon("Onagawa")) \
#          + from_latlon(op.lat("Hamaoka"), op.lon("Hamaoka")) \
#          + from_latlon(op.lat("Tohoku"), op.lon("Tohoku"))
# op.radii = [50e3, 50e3, 50e3, 50e3, 50e3, 50e3]

# Solve
q = tsunami(mesh, u0, eta0, b, BCs=BCs, f=f, nu=nu, op=op)
PETSc.Sys.Print("Objective functional value = %.4e" % q['J_h'])
PETSc.Sys.Print("CPU time = %.2f" % q['solverTime'])
