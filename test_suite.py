from firedrake import *

from utils.adaptivity import isotropicMetric, metricIntersection, metricGradation, normaliseIndicator, metricComplexity
from utils.options import GaussianOptions


mesh = UnitSquareMesh(30, 30)
P1 = FunctionSpace(mesh, "CG", 1)
op = GaussianOptions()
op.maxGrowth = 1.
op.nVerT = mesh.num_vertices()

f = Function(P1)
f.interpolate(Expression("pow(x[0], 2) + pow(x[1], 2) < 0.4 ? 1000. : 5."))
# f.assign(100.)
# f = normaliseIndicator(f, op=op)

M_f = isotropicMetric(f, invert=False, op=op)
mesh_f = AnisotropicAdaptation(mesh, M_f).adapted_mesh
File('plots/mesh_f.pvd').write(mesh_f.coordinates)
File('plots/metric_f.pvd').write(M_f)

g = Function(P1)
g.interpolate(Expression("pow(x[0] - 1., 2) + pow(x[1] - 1., 2) < 0.4 ? 1000. : 5."))
# g.assign(10000.)
# g = normaliseIndicator(g, op=op)

M_g = isotropicMetric(g, invert=False, op=op)
mesh_g = AnisotropicAdaptation(mesh, M_g).adapted_mesh
File('plots/mesh_g.pvd').write(mesh_g.coordinates)
File('plots/metric_g.pvd').write(M_g)

M_fg = metricIntersection(M_f, M_g)
# M_fg = metricIntersection(M_f, M_g, bdy=1)
# M_fg.dat.data[:] *= mesh.num_vertices() / metricComplexity(M_fg)
mesh_fg = AnisotropicAdaptation(mesh, M_fg).adapted_mesh
File('plots/mesh_fg.pvd').write(mesh_fg.coordinates)
File('plots/metric_fg.pvd').write(M_fg)

M_grad = metricGradation(M_fg, op=op)
mesh_grad = AnisotropicAdaptation(mesh, M_grad).adapted_mesh
File('plots/mesh_grad.pvd').write(mesh_grad.coordinates)
File('plots/metric_grad.pvd').write(M_grad)

print("Error norm of intersected and gradated metrics = ", errornorm(M_fg, M_grad))
