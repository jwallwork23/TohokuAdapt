## TohokuAdapt : anisotropic mesh adaptivity applied to the 2011 Tohoku tsunami ##

In this code, anisotropic mesh adaptivity is applied to solving the linearised shallow water equations in
[Firedrake][1], using the [PRAgMaTIc][2] toolbox. Code here is based on my [MRes project][3] at the Mathematics of
Planet Earth Centre for Doctoral Training [MPE CDT][4] at Imperial College London and University of Reading, and
integrates that work into the coastal, estuarine and ocean modelling solver provided by [Thetis][5].

Here you will find:
* A ``utils`` directory, containing the necessary functions for implementation of isotropic and anisotropic mesh
adaptivity:
    * Hessians and metrics are approximated using ``adaptivity``.
    * Coordinate transformations are achieved using ``conversion``.
    * (Local) error indicators are computed using ``error``.
    * Strong and weak forms of PDEs are available in ``forms``.
    * Interpolation of functions from an old mesh to a newly adapted mesh is achieved using ``interpolation``.
    * Meshes of the physical domain can be generated using ``mesh``.
    * Default parameters are specified using ``options``.
    * Time series data can be stored and plotted using ``storage``.
* A ``resources`` directory, containing bathymetry and coastline data for the ocean domain surrounding Fukushima. Mesh
files have been removed for copyright reasons, but may be made available upon request.
* Test problems ``advection-diffusion`` and ``shallow-water`` for advection diffusion and linear shallow water 
equations, respectively, defined on model domains. In both cases, there is functionality to run a fixed mesh, 'simple 
adaptive' or goal-oriented adaptive simulation.
* Simulations on a realistic domain, which build upon the test script codes and apply the methodology to the 2011 Tohoku
tsunami, which struck the Japanese coast at Fukushima and caused much destruction. These include:
    * ``firedrake-tsunami``, which solves the shallow water equations (1) on a fixed mesh, (2) using 'simple` mesh
     adaptivity and (3) using goal-oriented mesh adaptivity. The approach used encorporates automated differentiation 
     techniques to generate adjoint data in the discrete sense. The error indicators considered are formed of the 
     elementwise product of the primal residual and the adjoint solution.
    * ``thetis-tsunami``, which provides the same functionalities as ``firedrake-tsunami``, but integrated within the
    coastal and ocean modelling software provided by Thetis.
    * ``adjoint-based``, which solves the problem as guided by adjoint problem solution data. Errors are estimated using
    the product of the primal and dual solution fields, as in Davis and LeVeque 2016. In this case adjoint data is 
    obtained by solving the continuous adjoint shallow water equations. These data are used to guide the adaptive
    process in the sense that regions of significance can be attained by taking elementwise maxima over time windows.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[3]: https://github.com/jwallwork23/MResProject "MRes project"
[4]: http://mpecdt.org "MPE CDT"
[5]: http://thetisproject.org/index.html "Thetis"