## TohokuAdapt : anisotropic mesh adaptivity applied to the 2011 Tohoku tsunami ##

In this code, anisotropic mesh adaptivity is applied to solving the linearised shallow water equations in
[Firedrake][1], using the [PRAgMaTIc][2] toolbox. Code here is based on my [MRes project][3] at the Mathematics of
Planet Earth Centre for Doctoral Training [MPE CDT][4] at Imperial College London and University of Reading, and
integrates that work into the coastal, estuarine and ocean modelling solver provided by [Thetis][5].

Here you will find:
* A ``utils`` directory, containing the necessary functions for implementation of isotropic and anisotropic mesh
adaptivity:
    * Hessians and metrics are approximated using ``adaptivity``.
    * Bootstrapping for a near-optimal initial mesh is achieved using ``bootstrapping``.
    * Coordinate transformations are achieved using ``conversion``.
    * (Local) error indicators are computed using ``error``.
    * Strong and weak forms of PDEs are available in ``forms``.
    * Interpolation of functions from an old mesh to a newly adapted mesh is achieved using ``interpolation``.
    * Meshes of the physical domain can be generated using ``mesh``.
    * Options selections and statement printing is achieved using ``misc``.
    * Default parameters are specified using ``options``.
    * Time series data can be stored and plotted using ``timeseries``.
* A ``resources`` directory, containing bathymetry and coastline data for the ocean domain surrounding Fukushima. Mesh
files have been removed for copyright reasons, but may be made available upon request.
* Models on a realistic domain, which build upon the test script codes and apply the methodology to the 2011 Tohoku
tsunami, which struck the Japanese coast at Fukushima and caused much destruction. These include:
    * ``firedrake-tsunami``, which solves the shallow water equations using the following meshing strategies:
        * fixed meshes of various levels of refinement, generated using [QMESH][6];
        * hessian based error estimate;
        * explicit a posteriori error estimator based on the shallow water residual;
        * 'domain of dependence' type estimator as used in Davis and LeVeque 2016;
        * goal-oriented mesh adaptivity, weighting the residual by adjoint solution data.
    The latter two approaches encorporate automated differentiation techniques to generate adjoint data in the discrete 
    sense.
    * ``thetis-tsunami``, which provides the same functionalities as ``firedrake-tsunami``, but integrated within the
    coastal and ocean modelling software provided by Thetis.
* Test problems ``advection-diffusion``, ``shallow-water`` and ``rossby-wave`` for advection diffusion and linearised
 non-rotational and rotational shallow water equations, respectively, defined on quadrilateral model domains. In each 
 case, there is functionality to run fixed mesh or mesh adaptive simulations.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[3]: https://github.com/jwallwork23/MResProject "MRes project"
[4]: http://mpecdt.org "MPE CDT"
[5]: http://thetisproject.org/index.html "Thetis"
[6]: http://www.qmesh.org "QMESH"