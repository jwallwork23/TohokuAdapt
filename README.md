## TohokuAdapt : anisotropic mesh adaptivity applied to the 2011 Tohoku tsunami ##

In this code, anisotropic mesh adaptivity is applied to solving the linearised shallow water equations in
[Firedrake][1], using [PRAgMaTIc][2] to enable mesh adaptivity. Code builds upon my [MRes project][3] at the Mathematics
of Planet Earth Centre for Doctoral Training [MPE CDT][4] at Imperial College London and University of Reading. 
As part of my PhD research, that work is integrated into the coastal, estuarine and ocean modelling solver provided by 
[Thetis][5].

### Contents:
* A ``utils`` directory, containing the necessary functions for implementation of isotropic and anisotropic mesh
adaptivity:
    * Hessians and metrics are approximated using ``adaptivity``.
    * Callbacks for assembling objective functionals and extracting timeseries are provided in ``callbacks``.
    * Coordinate transformations are achieved using ``conversion``.
    * PDE strong and weak forms, (local) error indicators and analytic solutions are available in ``forms``.
    * Interpolation of functions from an old mesh to a newly adapted mesh is achieved using ``interpolation``.
    * Meshes of the physical domain can be generated using ``mesh``.
    * Some generic functions may be found in ``misc``.
    * Default parameters are specified using ``options``.
    * Time series and error estimate data can be stored and plotted using ``timeseries``.
    * Time stepping routines for the standalone model in ``timestepping``.
* Some basic tests for the mesh adaptivity functionalities above are provided in ``basic-tests.py``.
* A ``resources`` directory, containing bathymetry and coastline data for the ocean domain surrounding Fukushima. Mesh
files have been removed for copyright reasons, but may be generated in [QMESH][6] using the script ``utils/mesh``.
* Shallow water models ``firedrake-tsunami`` and ``thetis-tsunami`` for (1) a realistic domain, applied to the 2011 
Tohoku tsunami, which notably struck the Japanese coast at Fukushima; and (2) test scripts on quadrilateral model 
domains. Testing of linear vs. nonlinear and rotational vs. nonrotational models is achieved by ``model-verification``.
The following meshing strategies are implemented:
    * fixed meshes of various levels of refinement;
    * field, gradient and hessian based error estimates;
    * explicit and implicit a posteriori error estimators based on shallow water residuals;
    * a 'domain of dependence' type estimator as used in Davis and LeVeque 2016;
    * goal-oriented mesh adaptivity, weighting error estimates by adjoint solution data.
* Timeseries may be plotted using ``quick-timeseries``. Total variations between timeseries may be calculated using
``quick-difference``. Error curves and CPU timings may be plotted using  and ``quick-plot``.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[3]: https://github.com/jwallwork23/MResProject "MRes project"
[4]: http://mpecdt.org "MPE CDT"
[5]: http://thetisproject.org/index.html "Thetis"
[6]: http://www.qmesh.org "QMESH"