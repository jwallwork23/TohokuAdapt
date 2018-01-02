import matplotlib.pyplot as plt

import utils.bootstrapping as boot


problem = input("""Select from the following
      'advection-diffusion'
      'shallow-water'
      'rossby-wave'
      'firedrake-tsunami'
      'thetis-tsunami'\n""") or 'advection-diffusion'
(Js, nEls) = boot.bootstrap(problem)[1:]

plt.gcf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ylab = r'Objective functional'
if problem == 'advection-diffusion':
    plt.title(r'Advection-diffusion test case on $[0,4]\times[0,1]$')
    ylab += ' $J(\phi)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\phi\:\mathrm{d}x\mathrm{d}t$'
elif problem == 'shallow-water':
    plt.title(r'Non-rotating shallow water test case on $[0,2]\times[0,2]$')
    ylab += ' $J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
elif problem == 'rossby-wave':
    plt.title(r'Equatorial Rossby wave test case on $[-24,24]\times[-12,12]$')
    raise NotImplementedError
elif problem == 'firedrake-tsunami':
    plt.title(r'Tohoku tsunami problem solved using Firedrake')
    ylab += ' $J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
elif problem == 'thetis-tsunami':
    plt.title(r'Tohoku tsunami problem solved using Thetis')
    ylab += ' $J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
else:
    raise ValueError("Problem not recognised.")
plt.xlabel(r'\#Elements')
plt.ylabel(ylab)
if problem in ('firedrake-tsunami', 'thetis-tsunami'):
    plt.loglog(nEls, Js, basex=10, basey=10, marker='o')
else:
    plt.semilogx(nEls, Js, basex=10, marker='o')
plt.savefig("outdata/bootstrapping/" + problem + ".pdf", bbox_inches='tight')
plt.show()
