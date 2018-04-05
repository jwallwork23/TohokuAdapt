from firedrake import *

import matplotlib.pyplot as plt
import argparse

from utils.adaptivity import *
from utils.forms import indicator
from utils.mesh import meshStats
from utils.options import Options


def hessian():
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
        for hessMeth in ('dL2', 'parts'):
            op = Options(hessMeth=hessMeth)
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
                H = constructHessian(f, g=g, op=op)
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

                # Plot results
                di = "plots/adapt-tests/"
                File(di + "field_" + str(i) + ".pvd").write(f)
                File(di + "gradient_" + str(i) + ".pvd").write(g)
                File(di + "hessian_" + str(i) + ".pvd").write(H)
            plt.subplot(211)
            plt.loglog(nEls, grad_diff, label=hessMeth)
            plt.subplot(212)
            if subset:
                plt.loglog(nEls, hess_diff, label=hessMeth)
            else:
                plt.semilogx(nEls, hess_diff, label=hessMeth)
        plt.legend()
        plt.savefig("outdata/adapt-tests/subset="+str(subset)+"_function"+str(i)+".pdf")
        plt.clf()


def directionalRefine(eps=1e-4):
    for j in range(8):
        for dir in (0, 1):
            n = pow(2, j)
            mesh = UnitSquareMesh(n, n)
            op = Options(nVerT=meshStats(mesh)[1],
                         hmin=1e-8,
                         hmax=1)
            op.nVerT *= op.rescaling
            M_ = isotropicMetric(Function(FunctionSpace(mesh, space, 1)).interpolate(CellSize(mesh)), op=op)
            M = anisoRefine(M_, direction=dir)
            coords = AnisotropicAdaptation(mesh, M).adapted_mesh.coordinates.dat.data
            x = 0
            y = 0
            for i in range(len(coords)):
                if coords[i][0] < eps:
                    x += 1
                if coords[i][1] < eps:
                    y += 1
            ratio = y / x if dir == 0 else x / y

            print(ratio)    # TODO: Check what is going on here


# TODO: Multiple adaption tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test", help="Choose a test from {'Hessian', 'dirRefine'}")
    parser.add_argument("-subset", help="Toggle whether to calculate error over the whole domain or a subset thereof")
    parser.add_argument("-space", help="Toggle CG or DG space (default CG)")
    args = parser.parse_args()
    subset = args.subset if args.subset else True
    space = args.space if args.space else "CG"

    if args.test == 'Hessian':
        hessian()
    elif args.test == 'dirRefine':
        directionalRefine()
