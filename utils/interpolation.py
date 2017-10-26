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

# TODO: interpolation script for mixed spaces
