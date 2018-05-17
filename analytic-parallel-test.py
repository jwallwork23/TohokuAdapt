from firedrake import *
from firedrake.petsc import PETSc


mesh = SquareMesh(100, 100, 10, 10)
V = FunctionSpace(mesh, "DG", 1)
f = Function(V)
f.assign(1.)

def indicator(V):

    iA = Function(V, name="Region of interest")
    iA.interpolate(Expression(
        '(x[0] > %f - eps) && (x[0] < %f + eps) && (x[1] > %f - eps) && (x[1] < %f + eps) ? 1. : 0.' % (0., 2., 4., 6.),
        eps=1e-10))

    return iA

def objectiveSW(f):

    V = f.function_space()
    ks = Function(V)
    ks.assign(indicator(V))
    kt = Constant(1.)

    return assemble(kt * inner(ks, f) * dx)

PETSc.Sys.Print('Rank %d gives value %.4f' % (mesh.comm.rank, objectiveSW(f)), comm=COMM_SELF)
PETSc.Sys.Print('Global value %.4f' % objectiveSW(f), comm=COMM_WORLD)
