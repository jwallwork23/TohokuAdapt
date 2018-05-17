from firedrake import *
from firedrake.petsc import PETSc


mesh = SquareMesh(200, 200, 10, 10)
V = FunctionSpace(mesh, "DG", 1)
f = Function(V).assign(1.)

def indicator(V):

    iA = Function(V, name="Region of interest")
    iA.interpolate(Expression('(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (0., 2., 4., 6.)))

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