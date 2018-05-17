from firedrake import *
from firedrake.petsc import PETSc


mesh = SquareMesh(100, 100, 2, 2)
V = FunctionSpace(mesh, "DG", 1)
f = Function(V).assign(1.)

def indicator(V):

    xy = [0., 0.5, 0.5, 1.5]
    ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (xy[0], xy[1], xy[2], xy[3])
    iA = Function(V, name="Region of interest").interpolate(Expression(ind))

    return iA

def objectiveSW(f):
    """
    :param solver_obj: FlowSolver2d object.
    :return: objective functional value for callbacks.
    """
    V = f.function_space()
    ks = Function(V)
    ks.assign(indicator(V))
    kt = Constant(1.)

    return assemble(kt * inner(ks, f) * dx)


PETSc.Sys.Print('  rank %d gives value %.4f' % (mesh.comm.rank, objectiveSW(f)), comm=COMM_SELF)