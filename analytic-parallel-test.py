from firedrake import *
from firedrake.petsc import PETSc


mesh = SquareMesh(100, 100, 10, 10)
V = FunctionSpace(mesh, "DG", 1)
f = Function(V)
f.assign(1.)

def indicator(mesh):

    P0 = FunctionSpace(mesh, "DG", 0)
    x = SpatialCoordinate(mesh)
    iA = Function(P0, name="Region of interest")
    eps = 1e-10
    cond_x = And(gt(x[0], 0. - eps), lt(x[0], 2. + eps))
    cond_y = And(gt(x[1], 4. - eps), lt(x[1], 6. + eps))
    iA.interpolate(conditional(And(cond_x, cond_y), 1, 0))

    return iA

def objectiveSW(f):

    V = f.function_space()
    ks = indicator(V.mesh())
    kt = Constant(1.)

    return assemble(kt * inner(ks, f) * dx)

PETSc.Sys.Print('Rank %d gives value %.4f' % (mesh.comm.rank, objectiveSW(f)), comm=COMM_SELF)
PETSc.Sys.Print('Global value %.4f' % objectiveSW(f), comm=COMM_WORLD)
