from firedrake import *

import matplotlib.pyplot as plt

from utils.adaptivity import *
from utils.mesh import meshStats


grad_diff = []
hess_diff = []
nEls = []

for i in range(6):
    n = pow(2, i)
    mesh = SquareMesh(n, n, 2, 2)
    nEls.append(meshStats(mesh)[0])
    xy = Function(mesh.coordinates)
    xy.dat.data[:, :] -= [1, 1]
    mesh.coordinates.assign(xy)

    Vs = FunctionSpace(mesh, "CG", 1)
    Vv = VectorFunctionSpace(mesh, "CG", 1)
    Vt = TensorFunctionSpace(mesh, "CG", 1)

    # Establish function and (analytical) derivatives thereof
    f = Function(Vs).interpolate(Expression("x[0]**2 + x[1]**2"))   # TODO: Consider other functions (from Olivier?)
    grad_f = Function(Vv).interpolate(Expression(["2 * x[0]", "2 * x[1]"]))
    hess_f = Function(Vt).interpolate(Expression([[2, 0], [0, 2]]))

    # Compute these numerically and check similarity
    grad_diff.append(errornorm(grad_f, constructGradient(f)))
    hess_diff.append(errornorm(hess_f, constructHessian(f)))        # TODO: Test different projections

plt.plot([1, 2])
plt.subplot(211)
plt.plot(nEls, grad_diff)
plt.subplot(212)
plt.plot(nEls, hess_diff)
plt.savefig("plots/adapt-tests/numerical-vs-analytical.pdf")
plt.show()

# TODO: More tests
# TODO: Also test DG case