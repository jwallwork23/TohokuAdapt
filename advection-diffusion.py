from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

from utils.adaptivity import metricGradation, steadyMetric
from utils.interpolation import interp
from utils.misc import indicator
from utils.options import Options


def weakResidualAD(c, c_, w, op=Options(mode='advection-diffusion')):
    """
    :arg c: concentration solution at current timestep. 
    :arg c_: concentration at previous timestep.
    :arg w: wind field.
    :param nu: diffusivity parameter.
    :param op: Options parameter object.
    :return: weak residual for advection diffusion equation at current timestep.
    """
    if op.timestepper == 'CrankNicolson':
        cm = 0.5 * (c + c_)
    else:
        raise NotImplementedError
    ct = TestFunction(c.function_space())
    F = ((c - c_) * ct / Constant(op.dt) + inner(grad(cm), w * ct)) * dx
    F += Constant(op.viscosity) * inner(grad(cm), grad(ct)) * dx
    return F


def fixedMesh(n=3):
    op = Options(mode='advection-diffusion')
    forwardFile = File(op.di + "fixedMesh.pvd")

    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
    x, y = SpatialCoordinate(mesh)

    # Define FunctionSpaces and specify physical and solver parameters
    V = FunctionSpace(mesh, "CG", 1)
    w = Function(VectorFunctionSpace(mesh, "CG", 1), name='Wind field').interpolate(Expression([1, 0]))

    # Apply initial condition and define Functions
    ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
    phi = ic.copy(deepcopy=True)
    phi.rename('Concentration')
    phi_next = Function(V, name='Concentration next')
    nEle = mesh.num_cells()
    F = weakResidualAD(phi_next, phi, w, op=op)

    t = 0.
    cnt = 0
    quantities = {}
    fullTimer = 0.
    if op.plotpvd:
        forwardFile.write(phi, time=t)
    iA = indicator(mesh, xy=[3., 0.], radii=0.5, op=op)
    J_list = [assemble(iA * phi * dx)]
    while t < op.Tend:
        # Solve problem at current timestep
        solverTimer = clock()
        solve(F == 0, phi_next)
        solverTimer = clock() - solverTimer
        fullTimer += solverTimer
        phi.assign(phi_next)

        J_list.append(assemble(iA * phi * dx))

        if op.plotpvd & (cnt % op.ndump == 0):
            forwardFile.write(phi, time=t)
        print('t = %.3fs' % t)
        t += op.dt
        cnt += 1

    J_h = 0.
    for i in range(1, len(J_list)):
        J_h += 0.5 * (J_list[i] + J_list[i-1]) * op.dt

    quantities['meanElements'] = nEle
    quantities['solverTimer'] = fullTimer
    quantities['J_h'] = J_h

    return quantities

def hessianBased(n=3):
    op = Options(mode='advection-diffusion')
    forwardFile = File(op.di + "hessianBased.pvd")

    # Define problem
    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
    phi = ic.copy(deepcopy=True)
    phi.rename('Concentration')
    phi_next = Function(V, name='Concentration next')

    # Get adaptivity parameters
    nEle = mesh.num_cells()
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    op.nVerT = mesh.num_vertices() * op.vscale  # Target #Vertices

    # Initialise counters
    t = 0.
    cnt = 0
    adaptSolveTimer = 0.
    quantities = {}
    iA = indicator(mesh, xy=[3., 0.], radii=0.5, op=op)
    J_list = [assemble(iA * phi * dx)]
    while t <= op.Tend:
        adaptTimer = clock()
        for l in range(op.nAdapt):

            # Construct metric
            M = steadyMetric(phi, op=op)
            if op.gradate:
                metricGradation(mesh, M)

            # Adapt mesh and interpolate variables
            mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            phi = interp(mesh, phi)
            phi.rename("Concentration")
            V = FunctionSpace(mesh, "CG", 2)
            phi_next = Function(V)

            iA = indicator(mesh, xy=[3., 0.], radii=0.5, op=op)
            J_list.append(assemble(iA * phi * dx))

            # Re-establish bilinear form and set boundary conditions
            w = Function(VectorFunctionSpace(mesh, "CG", 1), name='Wind field').interpolate(Expression([1, 0]))
            F = weakResidualAD(phi_next, phi, w, op=op)

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle

        # Solve problem at current timestep
        solverTimer = clock()
        solve(F == 0, phi_next)
        phi.assign(phi_next)
        solverTimer = clock() - solverTimer

        # Print to screen, save data and increment counters
        print('t = %.3fs' % t)
        if op.plotpvd & (cnt % op.ndump == 0):
            forwardFile.write(phi, time=t)
        t += op.dt
        cnt += 1
        av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
        adaptSolveTimer += adaptTimer + solverTimer

    J_h = 0.
    for i in range(1, len(J_list)):
        J_h += 0.5 * (J_list[i] + J_list[i - 1]) * op.dt

    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['J_h'] = J_h

    return quantities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'hessianBased', 'DWP', 'DWR'} (default 'fixedMesh')")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    args = parser.parse_args()

    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    solvers = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased}

    for i in resolutions:
        q = solvers[args.a]
        print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs"
              % (i, q['meanElements'], q['J_h'], q['solverTimer']))

