from firedrake import *

import matplotlib.pyplot as plt
import argparse
import datetime
import numpy as np

from utils.adaptivity import *
from utils.interpolation import interp
from utils.misc import indicator
from utils.options import Options


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')

functions = ["x[0]*x[0] + x[1]*x[1]", "sin(pi*x[0])*sin(pi*x[1])"]
gradients = [{0: "2 * x[0]", 1: "2 * x[1]"},
             {0: "pi*cos(pi*x[0])*sin(pi*x[1])", 1: "pi*sin(pi*x[0])*cos(pi*x[1])"}]
hessians = [[[2, 0], [0, 2]],
            [["-pi*pi*sin(pi*x[0])*sin(pi*x[1])", "pi*pi*cos(pi*x[0])*cos(pi*x[1])"],
             ["pi*pi*cos(pi*x[0])*cos(pi*x[1])", "-pi*pi*sin(pi*x[0])*sin(pi*x[1])"]]]

def integrate(func, xy):
    if func == 0:
        return (xy[1]**3-xy[0]**3)*(xy[3]-xy[2])/3. + (xy[3]**3-xy[2]**3)*(xy[1]-xy[0])/3.
    elif func == 1:
        return (cos(pi * xy[1]) - cos(pi * xy[0])) * (cos(pi * xy[3]) - cos(pi * xy[2])) / (pow(pi, 2))
    else:
        raise NotImplementedError

styles = {'dL2': 'x', 'parts': 'o'}
labels = {'dL2': r'Double $\mathcal{L}_2$ projection', 'parts': r'Integration by parts'}
titles = (r'$f_1(x,y)=x^2+y^2$', r'$f_2(x,y)=\sin(\pi x)\sin(\pi y)$')


def hessian(subset, space):
    plt.plot([1, 2])
    plt.subplot(211)
    plt.ylabel("L2 error in gradient")
    plt.subplot(212)
    plt.xlabel("Element count")
    plt.ylabel("L2 error in Hessian")
    for i in range(len(functions)):
        for hessMeth in ('dL2', 'parts'):
            op = Options(mode=None,
                         approach='hessianBased')
            op.hessMeth = hessMeth
            op.di = 'plots/adapt-tests/'
            grad_diff = []
            hess_diff = []
            nEls = []
            for j in range(2, 8):
                n = pow(2, j)
                mesh = SquareMesh(n, n, 2, 2)
                nEle = mesh.num_cells()
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
                    iA = indicator(mesh, xy=[-0.8, -0.4, -0.2, 0.2], op=op)
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
            plt.loglog(nEls, grad_diff, label=labels[hessMeth], marker=styles[hessMeth])
            plt.title(titles[i])
            plt.subplot(212)
            if subset:
                plt.loglog(nEls, hess_diff, label=labels[hessMeth], marker=styles[hessMeth])
            else:
                plt.semilogx(nEls, hess_diff, label=labels[hessMeth], marker=styles[hessMeth])
        plt.legend()
        plt.savefig("outdata/adapt-tests/subset="+str(subset)+"_function"+str(i)+'_'+date+".pdf")
        plt.clf()


def adapts(scale, space, indy):
    if indy == 'aligned':
        region = [-0.82, -0.44, -0.21, 0.33]    # Test 1: a box which lines up with the grid
        r = None
    elif indy == 'misaligned':
        region = [-0.82, -0.44, -0.21, 0.33]    # Test 2: a box which does not line up with the grid
        r = None
    else:
        region = [-0.6, 0.1]                    # Test 3: a disc
        r = 0.2
    op = Options(mode=None, approach='hessianBased')
    op.hmin = 1e-10
    op.hmax = 1
    for i in range(len(functions)):
        J = integrate(i, xy=region)
        for nAdapt in range(1, 5):
            op.di = 'plots/adapt-tests/'
            adapt_diff = []
            fixed_diff = []
            nEls = []
            inEls = []
            for j in range(2, 8):
                n = pow(2, j)
                mesh = SquareMesh(n, n, 2, 2)
                if scale:
                    op.nVerT = op.rescaling * mesh.num_vertices()
                inEls.append(mesh.num_cells())
                xy = Function(mesh.coordinates)
                xy.dat.data[:, :] -= [1, 1]
                mesh.coordinates.assign(xy)
                Vs = FunctionSpace(mesh, space, 1)

                # Establish function and (analytical) functional value
                f = Function(Vs).interpolate(Expression(functions[i]))
                if nAdapt == 1:
                    iA = indicator(mesh, xy=region, radius=r, op=op)
                    J_fixed = assemble(f * iA * dx)
                    fixed_diff.append(np.abs((J - J_fixed) / J))

                # Adapt mesh
                for k in range(nAdapt):
                    M = steadyMetric(f, op=op)
                    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                    f = interp(mesh, f)
                nEle = mesh.num_cells()
                nEls.append(nEle)

                # Calculate difference in functional approximation
                iA = indicator(mesh, xy=region, radius=r, op=op)
                J_adapt = assemble(f * iA * dx)
                adapt_diff.append(np.abs((J - J_adapt) / J))
                print("Function %d, run %d, %s space, %d elements: Functional = %.4f, Error = %.4f)"
                      % (i, j, space, nEle, J_adapt, adapt_diff[-1]))

            if nAdapt == 1:
                plt.loglog(inEls, fixed_diff, label="Fixed mesh", marker='x')
            label = "%d adaptations" % nAdapt
            plt.loglog(nEls if scale else inEls, adapt_diff, label=label, marker='o')
        plt.title(titles[i])
        plt.xlabel("Element count" if scale else "Inital element count")
        plt.ylabel("Relative error in functional")
        plt.legend()
        plt.savefig("outdata/adapt-tests/test="+str(indy)+"_scale="+str(scale)+'_function'+str(i)+'_'+date+".pdf")
        plt.clf()


def directionalRefine(eps=1e-4):
    for j in range(8):
        for dir in (0, 1):
            n = pow(2, j)
            mesh = UnitSquareMesh(n, n)
            op = Options(mode=None, hmin=1e-8, hmax=1)
            op.nVerT = mesh.num_vertices() * op.rescaling
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test", help="Choose a test from {'Hessian', 'dirRefine'}")
    parser.add_argument("-subset", help="Toggle whether to calculate error over the whole domain or a subset thereof")
    parser.add_argument("-space", help="Toggle CG or DG space (default CG)")
    parser.add_argument("-scale", help="Toggle scaling of vertex count")
    parser.add_argument("-i", help="Type of indicator function, from {'aligned', 'misaligned', 'disc'}")
    args = parser.parse_args()
    subset = bool(args.subset)
    space = args.space if args.space else "CG"
    scale = bool(args.scale)
    assert args.space in ("CG", "DG")
    indy = 'aligned' if args.i is None else args.i
    assert indy in ('aligned', 'misaligned', 'disc')
    print(subset, space, indy)

    if args.test == 'Hessian':
        hessian(subset, space)
    elif args.test == 'dirRefine':
        directionalRefine()
    elif args.test == 'nAdapt':
        adapts(scale, space, indy)
