from firedrake import *

import utils.adaptivity as adap
import utils.interpolation as inte


# Set up a uniform mesh and define a scalar function thereupon:
mesh = SquareMesh(100, 100, 2, 2)
x, y = SpatialCoordinate(mesh)
x = x - 1
y = y - 1
V = FunctionSpace(mesh, 'CG', 1)
f = Function(V)
f.interpolate(0.1 * sin(50 * x) + atan(0.1 / (sin(5 * y) - 2 * x)))
File('plots/tests/pre-adapt.pvd').write(f)

# Create a metric to guide the adaptivity process:
W = TensorFunctionSpace(mesh, 'CG', 1)
H = adap.constructHessian(mesh, W, f)
M = adap.computeSteadyMetric(mesh, W, H, f, h_min=0.0001, a=100.)

# Adapt the mesh according to this metric:
adaptor = AnisotropicAdaptation(mesh, M)
newmesh = adaptor.adapted_mesh

# Interpolate the scalar field onto the new mesh and plot:
g = inte.interp(newmesh, f)[0]
File('plots/tests/post-adapt.pvd').write(g)
