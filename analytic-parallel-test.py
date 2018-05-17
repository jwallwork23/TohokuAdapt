from thetis import *
from firedrake.petsc import PETSc

from utils.misc import indicator
from utils.options import Options


mesh = SquareMesh(100, 100, 2 * pi, 2 * pi)
V = FunctionSpace(mesh, "DG", 1)
f = Function(V).assign(1.)

def objectiveSW(f):
    """
    :param solver_obj: FlowSolver2d object.
    :return: objective functional value for callbacks.
    """
    V = f.function_space()
    ks = Function(V)
    ks.assign(indicator(V, op=Options(mode='shallow-water')))
    kt = Constant(1.)

    return assemble(kt * inner(ks, f) * dx)


PETSc.Sys.Print('  rank %d owns %d elements and can access %d vertices. Gives value %.4f' \
                % (mesh.comm.rank, mesh.num_cells(), mesh.num_vertices(), objectiveSW(f)),
                comm=COMM_SELF)