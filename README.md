## TohokuAdapt : anisotropic mesh adaptivity applied to the 2011 Tohoku tsunami ##

In this code, anisotropic mesh adaptivity is applied to solving the linearised shallow water equations in Firedrake.

Here you will find:
* A ``utils`` directory, containing the necessary functions for implementation of isotropic and anisotropic mesh
adaptivity:
    * Hessians and metrics can be approximated using ``adaptivity``.
    * Coordinate transformations are achieved using ``conversion``.
    * Meshes and initial conditions on the realistic ocean domain are generated using ``domain``.
    * Interpolation of functions from an old mesh to a newly adapted mesh is achieved using ``interpolation``.
    * Default parameters are specified in ``options``.
    * Time series data can be stored and plotted using ``storage``.
* A ``resources`` directory, containing bathymetry and coastline data for the ocean domain surrounding Fukushima. Mesh
files have been removed for copyright reasons, but may be made available upon request.
* Simulations on a realistic domain, which build upon the test script codes and apply the methodology to the 2011 Tohoku
tsunami, which struck the Japanese coast at Fukushima and caused much destruction. These include:
    * ``fixedMesh``, which solves the problem without mesh adaptivity.
    * ``simpleAdapt``, which solves the problem using anisotropic mesh adaptivity.
    * ``adjointBased``, which solve the problem as guided by adjoint problem solution data.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.