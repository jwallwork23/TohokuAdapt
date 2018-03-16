from thetis import *
from thetis.field_defs import field_metadata
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt


assert (float(physical_constants['g_grav'].dat.data) == 9.81)

def solverSW(startRes, op=opt.Options()):

    # Establish Mesh
    mesh_H, eta0, b = msh.TohokuDomain(startRes, wd=op.wd)
    P1 = FunctionSpace(mesh_H, "CG", 1)