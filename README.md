## TohokuAdapt : anisotropic mesh adaptivity applied to the 2011 Tohoku tsunami ##

In this code, anisotropic mesh adaptivity is applied to solving the nonlinear shallow water equations in the coastal, 
estuarine and ocean modelling solver provided by [Thetis][1]. The Thetis project is built upon the [Firedrake][2]
project, which enables efficient FEM solution in Python by automatic generation of C code. Anisotropic mesh adaptivity
is achieved using [PRAgMaTIc][3]. Code builds upon my [MRes project][4] at the Mathematics of Planet Earth Centre for 
Doctoral Training ([MPE CDT][5]) at Imperial College London and University of Reading.

### Contents:
* A ``utils`` directory, containing the necessary functions for implementation of isotropic and anisotropic mesh
adaptivity:
    * Hessians and metrics are approximated using ``adaptivity``.
    * Callbacks for assembling objective functionals and extracting timeseries are provided in ``callbacks``.
    * Coordinate transformations are achieved using ``conversion``.
    * Interpolation of functions from an old mesh to a newly adapted mesh is achieved using ``interpolation``.
    * Some generic functions may be found in ``misc``.
    * Default parameters are specified using ``options``.
    * Meshes of the physical domain, bathymetry and initial and boundary conditions are generated using ``setup``.
    * Fixed mesh and mesh adaptive solvers can be found in ``ad_solvers`` and ``sw_solvers``, for advection-diffusion
    and shallow water, respectively.
    * Time series and error estimate data can be stored and plotted using ``timeseries``.
* Some basic tests for the mesh adaptivity functionalities above are provided in ``basic-tests.py``.
* A ``resources`` directory, containing bathymetry and coastline data for the ocean domain surrounding Fukushima. Mesh
files have been removed for copyright reasons, but may be generated in [QMESH][6] using the script ``utils/mesh``.
* Shallow water model ``tsunami`` for (1) a realistic domain, applied to the 2011 
Tohoku tsunami, which notably struck the Japanese coast at Fukushima; and (2) test scripts on quadrilateral model 
domains with flat bathymetry. Testing of rotational vs. non-rotational models is achieved by ``model-verification``.
The following meshing strategies are implemented, with a number of other approaches in progress:
    * fixed meshes of various levels of refinement;
    * Hessian based error estimates;
    * a 'domain of dependence' type estimator as used in Davis and LeVeque 2016;
    * goal-oriented mesh adaptivity, weighting error estimates by adjoint solution data.
* Given data generated using the above methods, we may
    * plot timeseries using ``quick-timeseries``;
    * integrate timeseries using ``quick-integrate``; 
    * plot error curves and CPU timings against element count using ``quick-plot``.
* A more complex region of interest is considered in ``demo-script``, involving a union of disc regions.
* An additional test case is provided by the ``advect`` script.
    
### User instructions

Download the [Firedrake][1] install script, set ``export PETSC_CONFIGURE_OPTIONS=“download-pragmatic=1"`` and install 
with option parameters ``--install pyadjoint`` and ``--install thetis``. Fetch and checkout the remote branches 
* ``https://github.com/taupalosaurus/firedrake`` for firedrake;
* ``https://bitbucket.org/dolfin-adjoint/pyadjoint/branch/linear-solver`` for pyadjoint;
* ``https://github.com/thetisproject/thetis/tree/goal-based-adaptation`` for thetis.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://thetisproject.org/index.html "Thetis"
[2]: http://firedrakeproject.org/ "Firedrake"
[3]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[4]: https://github.com/jwallwork23/MResProject "MRes project"
[5]: http://mpecdt.org "MPE CDT"
[6]: http://www.qmesh.org "QMESH"