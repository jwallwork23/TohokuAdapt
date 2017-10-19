from firedrake import *


def interp(mesh, *fields):
    """
    Transfer solution fields from the old mesh to the new mesh. Based around the function ``transfer_solution`` by
    Nicolas Barral, 2017.
    
    :param mesh: new mesh onto which fields are to be interpolated.
    :param fields: tuple of functions defined on the old mesh that one wants to transfer
    :return: interpolated fields.
    """
    dim = mesh._topological_dimension
    assert dim == 2                     # 3D implementation not yet considered

    fields_new = ()
    for f in fields:
        V_new = FunctionSpace(mesh, f.function_space().ufl_element())
        f_new = Function(V_new)
        notInDomain = []

        if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
            coords = mesh.coordinates.dat.data                                      # Vertex/node coords
        elif f.ufl_element().family() == 'Lagrange':
            degree = f.ufl_element().degree()
            C = VectorFunctionSpace(mesh, 'CG', degree)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(mesh.coordinates)
            coords = interp_coordinates.dat.data                                    # Node coords (NOT just vertices)
        elif f.ufl_element().family() == 'Discontinuous Lagrange':
            degree = f.ufl_element().degree()
            C = VectorFunctionSpace(mesh, 'DG', degree)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(mesh.coordinates)
            coords = interp_coordinates.dat.data
        else:
            raise NotImplementedError('Interpolation not currently supported on requested field type.')

        try:
            f_new.dat.data[:] = f.at(coords)
        except PointNotInDomainError:
            # print('#### Points not in domain! Commence attempts by increasing tolerances')

            # Establish which vertices fall outside the domain:
            for x in range(len(coords)):
                try:
                    val = f.at(coords[x])
                except PointNotInDomainError:
                    val = 0.
                    notInDomain.append(x)
                finally:
                    f_new.dat.data[x] = val
        eps = 1e-6                              # Tolerance to be increased
        while len(notInDomain) > 0:
            # print('#### Points not in domain: %d / %d' % (len(notInDomain), len(coords)),)
            eps *= 10
            # print('...... Trying tolerance = ', eps)
            for x in notInDomain:
                try:
                    val = f.at(coords[x], tolerance=eps)
                except PointNotInDomainError:
                    val = 0.
                finally:
                    f_new.dat.data[x] = val
                    notInDomain.remove(x)
            if eps >= 1e8:
                print('#### Playing with epsilons failed. Abort.')
                exit(23)

        fields_new += (f_new,)
    return fields_new


def interpTaylorHood(mesh, u, u_, eta, eta_, b):
    """
    Transfer a mixed shallow water Taylor-Hood solution pair from the old mesh to the new mesh. Based around the
    function ``transfer_solution`` by Nicolas Barral, 2017.
    
    :param mesh: new mesh onto which fields are to be interpolated.
    :param u: P2 velocity field at the current timestep.
    :param u_: P2 velocity field at the previous timestep.
    :param eta: P1 free surface displacement at the current timestep.
    :param eta_: P1 free surface displacement at the previous timestep.
    :param b: bathymetry field.
    :return: interpolated individual and mixed variable pairs, along with the associated Taylor-Hood space.
    """

    dim = mesh._topological_dimension
    assert (dim == 2)                   # 3D implementation not yet considered

    W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
    qnew = Function(W)
    unew, etanew = qnew.split()
    q_new = Function(W)
    u_new, eta_new = q_new.split()
    bnew = Function(W.sub(1))
    notInDomain = []

    P1coords = mesh.coordinates.dat.data
    P2coords = Function(W.sub(0)).interpolate(mesh.coordinates).dat.data

    # Establish which vertices fall outside the domain:
    for x in range(len(P2coords)):
        try:
            valu = u.at(P2coords[x])
            valu_ = u_.at(P2coords[x])
        except PointNotInDomainError:
            valu = [0., 0.]
            valu_ = [0., 0.]
            notInDomain.append(x)
        finally:
            unew.dat.data[x] = valu
            u_new.dat.data[x] = valu_

    eps = 1e-6  # For playing with epsilons
    while len(notInDomain) > 0:
        # print('#### Points not in domain for P2 space: %d / %d' % (len(notInDomain), len(P2coords)),)
        eps *= 10
        # print('...... Trying epsilon = ', eps)
        for x in notInDomain:
            try:
                valu = u.at(P2coords[x], tolerance=eps)
                valu_ = u_.at(P2coords[x], tolerance=eps)
            except PointNotInDomainError:
                valu = [0., 0.]
                valu_ = [0., 0.]
            finally:
                unew.dat.data[x] = valu
                u_new.dat.data[x] = valu_
                notInDomain.remove(x)
        if eps > 1e8:
            print('#### Playing with epsilons failed. Abort.')
            exit(23)
    assert (len(notInDomain) == 0)  # All nodes should have been brought back into the domain

    # Establish which vertices fall outside the domain:
    for x in range(len(P1coords)):
        try:
            vale = eta.at(P1coords[x])
            vale_ = eta_.at(P1coords[x])
            valb = b.at(P1coords[x])
        except PointNotInDomainError:
            vale = 0.
            vale_ = 0.
            valb = 0.
            notInDomain.append(x)
        finally:
            etanew.dat.data[x] = vale
            eta_new.dat.data[x] = vale_
            bnew.dat.data[x] = valb

    eps = 1e-6  # For playing with epsilons
    while len(notInDomain) > 0:
        # print('#### Points not in domain for P1 space: %d / %d' % (len(notInDomain), len(P1coords)),)
        eps *= 10
        # print('...... Trying epsilon = ', eps)
        for x in notInDomain:
            try:
                vale = eta.at(P1coords[x], tolerance=eps)
                vale_ = eta_.at(P1coords[x], tolerance=eps)
                valb = b.at(P1coords[x], tolerance=eps)
            except PointNotInDomainError:
                vale = 0.
                vale_ = 0.
                valb = 0.
            finally:
                etanew.dat.data[x] = vale
                eta_new.dat.data[x] = vale_
                bnew.dat.data[x] = valb
                notInDomain.remove(x)
        if eps > 1e8:
            print('#### Playing with epsilons failed. Abort.')
            exit(23)

    return unew, u_new, etanew, eta_new, qnew, q_new, bnew, W
