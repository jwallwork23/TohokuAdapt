from firedrake import *


def setupSSPRK(B, L1, q):
    """
    Three-stage Strong-Stablility-Preserving Runge-Kutta timestepping, using code documented in 
    https://firedrakeproject.org/demos/DG_advection.py.html.
    
    :arg B: bilinear form, constructed using a TrialFunction.
    :arg L1: linear form.
    :arg q: prognostic variable of L1.
    :return: Expressions and Functions to solve to obtain the prognostic variable at next timestep.
    """
    # Set up forms for each stage
    V = q.function_space()
    q1 = Function(V)
    q2 = Function(V)
    L2 = replace(L1, {q: q1})
    L3 = replace(L1, {q: q2})
    dq = Function(V)    # Temporary variable to hold increments.

    # Define solver parameters
    params = {'ksp_type': 'preonly',
              'pc_type': 'bjacobi',     # Allows the code to be executed in parallel without any further changes.
              'sub_pc_type': 'ilu'}     # Used since DG mass matrices are block-diagonal.
    prob1 = LinearVariationalProblem(B, L1, dq)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = LinearVariationalProblem(B, L2, dq)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = LinearVariationalProblem(B, L3, dq)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

    return solv1, solv2, solv3, q1, q2, dq


def incrementSSPRK(solv1, solv2, solv3, q, q1, q2, dq):
    """
    Increment using ``setupSSPRK`` above.
    
    :param solv1: first solver object.
    :param solv2: second solver object.
    :param solv3: third solver object.
    :param q: prognostic variable.
    :param q1: first intermediary variable.
    :param q2: second intermediary variable.
    :param dq: incremental variable. 
    """
    solv1.solve()
    q1.assign(q + dq)

    solv2.solve()
    q2.assign(0.75 * q + 0.25 * (q1 + dq))

    solv3.solve()
    q.assign((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dq))
