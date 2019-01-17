outdirs:
	@echo "Building outdata directories"
	@mkdir outdata
	@mkdir outdata/Tohoku
	@mkdir outdata/Tohoku/FixedMesh
	@mkdir outdata/Tohoku/HessianBased
	@mkdir outdata/Tohoku/DWP
	@mkdir outdata/Tohoku/DWR
	@mkdir outdata/AdvectionDiffusion
	@mkdir outdata/AdvectionDiffusion/FixedMesh
	@mkdir outdata/AdvectionDiffusion/HessianBased
	@mkdir outdata/AdvectionDiffusion/DWP
	@mkdir outdata/AdvectionDiffusion/DWR
	@mkdir resources/meshes

clean:
	rm -Rf outdata
