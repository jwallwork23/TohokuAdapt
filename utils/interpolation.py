from firedrake import *

from .conversion import OutOfRangeError


__all__ = ["interp", "mixedPairInterp"]


def interp(mesh, *fields):
    """
    Transfer solution fields from the old mesh to the new mesh. Based around the function ``transfer_solution`` by
    Nicolas Barral, 2017.

    :arg mesh: new mesh onto which fields are to be interpolated.
    :arg fields: tuple of functions defined on the old mesh that one wants to transfer
    :return: interpolated fields.
    """
    dim = mesh._topological_dimension
    try:
        assert dim == 2
    except:
        raise NotImplementedError('3D implementation not yet considered.')
    fields_new = ()
    for f in fields:
        V_new = FunctionSpace(mesh, f.function_space().ufl_element())
        f_new = Function(V_new)
        notInDomain = []
        if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
            coords = mesh.coordinates.dat.data  # Vertex/node coords
        elif f.ufl_element().family() == 'Lagrange':
            degree = f.ufl_element().degree()
            C = VectorFunctionSpace(mesh, 'CG', degree)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(mesh.coordinates)
            coords = interp_coordinates.dat.data  # Node coords (NOT just vertices)
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
        # Establish which vertices fall outside the domain  # TODO: Figure out how to do this in a less hacky way
        except PointNotInDomainError:
            for x in range(len(coords)):
                try:
                    val = f.at(coords[x])
                except PointNotInDomainError:
                    val = 0.
                    notInDomain.append(x)
                finally:
                    f_new.dat.data[x] = val
        eps = 1e-6  # Tolerance to be increased
        while len(notInDomain) > 0:
            eps *= 10
            for x in notInDomain:
                try:
                    val = f.at(coords[x], tolerance=eps)
                except PointNotInDomainError:
                    val = 0.
                finally:
                    f_new.dat.data[x] = val
                    notInDomain.remove(x)
            if eps >= 1e8:
                raise OutOfRangeError('Playing with epsilons failed. Abort.')
        fields_new += (f_new,)
    return fields_new


def mixedPairInterp(mesh, V, *fields):
    """
    Interpolate mixed function space pairs onto a new mesh.

    :arg mesh: new mesh to be interpolated onto.
    :arg V: mixed function space defined on new mesh, with same type as that on which fields are defined.
    :arg fields: fields to be interpolated.
    :return: interpolated function pairs.
    """
    fields_new = ()
    for q in fields:
        p = Function(V)
        p0, p1 = p.split()
        q0, q1 = q.split()
        q0, q1 = interp(mesh, q0, q1)
        p0.assign(q0), p1.assign(q1)
        fields_new += (p,)
    return fields_new