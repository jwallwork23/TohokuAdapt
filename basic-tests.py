from firedrake import *

import matplotlib.pyplot as plt

from utils.adaptivity import *
from utils.forms import indicator
from utils.mesh import meshStats


subset = True   # TODO: argparse this up
space = "CG"

di = "plots/adapt-tests/"

functions = ["x[0]*x[0] + x[1]*x[1]", "sin(pi*x[0])*sin(pi*x[1])"]
gradients = [{0: "2 * x[0]", 1: "2 * x[1]"},
             {0: "pi*cos(pi*x[0])*sin(pi*x[1])", 1: "pi*sin(pi*x[0])*cos(pi*x[1])"}]
hessians = [[[2, 0], [0, 2]],
            [["-pi*pi*sin(pi*x[0])*sin(pi*x[1])", "pi*pi*cos(pi*x[0])*cos(pi*x[1])"],
             ["pi*pi*cos(pi*x[0])*cos(pi*x[1])", "-pi*pi*sin(pi*x[0])*sin(pi*x[1])"]]]

plt.plot([1, 2])
plt.subplot(211)
plt.ylabel("L2 error in gradient")
plt.subplot(212)
plt.xlabel("Element count")
plt.ylabel("L2 error in Hessian")
for i in range(len(functions)):
    grad_diff = []
    hess_diff = []
    nEls = []
    for j in range(2, 8):
        n = pow(2, j)
        mesh = SquareMesh(n, n, 2, 2)
        nEle = meshStats(mesh)[0]
        nEls.append(nEle)
        xy = Function(mesh.coordinates)
        xy.dat.data[:, :] -= [1, 1]
        mesh.coordinates.assign(xy)

        Vs = FunctionSpace(mesh, space, 1)
        Vv = VectorFunctionSpace(mesh, space, 1)
        Vt = TensorFunctionSpace(mesh, "CG", 1)

        # Establish function and (analytical) derivatives thereof
        f = Function(Vs).interpolate(Expression(functions[i]))
        grad_f = Function(Vv).interpolate(Expression([gradients[i][0], gradients[i][1]]))
        hess_f = Function(Vt).interpolate(Expression(hessians[i]))

        # Compute these numerically and check similarity
        g = constructGradient(f)
        H = constructHessian(f, g=g)
        if subset:
            iA = indicator(Vs, 'basic')
            grad_err = norm(iA * (grad_f - g)) / norm(iA * grad_f)
            hess_err = norm(iA * (hess_f - H)) / norm(iA * hess_f)
        else:
            grad_err = errornorm(grad_f, g) / norm(grad_f)
            hess_err = errornorm(hess_f, H) / norm(hess_f)
        grad_diff.append(grad_err)
        hess_diff.append(hess_err)
        print("Function %d, run %d, %s space, %d elements: Gradient error = %.4f, Hessian error = %.4f"
              % (i, j, space, nEle, grad_err, hess_err))

        File(di + "field_" + str(i) + ".pvd").write(f)
        File(di + "gradient_" + str(i) + ".pvd").write(g)
        File(di + "hessian_" + str(i) + ".pvd").write(H)

        plt.subplot(211)
        plt.loglog(nEls, grad_diff)
        plt.subplot(212)
        if subset:
            plt.loglog(nEls, hess_diff)
        else:
            plt.semilogx(nEls, hess_diff)
    # plt.legend()
    plt.subplot(211)
    # plt.legend()
    plt.savefig("outdata/adapt-tests/numerical-vs-analytical_subset="+str(subset)+"_function"+str(i)+".pdf")
    plt.clf()

# TODO: More tests