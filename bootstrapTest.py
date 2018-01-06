import matplotlib.pyplot as plt

import utils.bootstrapping as boot


problem = input("""Select from the following
      'advection-diffusion'
      'shallow-water'
      'rossby-wave'
      'firedrake-tsunami'
      'thetis-tsunami'\n""") or 'advection-diffusion'

plt.gcf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ylab1 = r'Objective functional '
ylab2 = r'Run time'
if problem == 'advection-diffusion':
    title = r'Advection-diffusion test case on $[0,4]\times[0,1]$'
    ylab1 += '$J(\phi)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\phi\:\mathrm{d}x\mathrm{d}t$'
    tol = 1e-2
elif problem == 'shallow-water':
    title = r'Non-rotating shallow water test case on $[0,2]\times[0,2]$'
    ylab1 += '$J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
    tol = 1e-2
elif problem == 'rossby-wave':
    title = r'Equatorial Rossby wave test case on $[-24,24]\times[-12,12]$'
    raise NotImplementedError
elif problem == 'firedrake-tsunami':
    title = r'Tohoku tsunami problem solved using Firedrake'
    ylab1 += '$J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
    tol = 1.5e10      # Note J ~ 2.4e13
elif problem == 'thetis-tsunami':
    title = r'Tohoku tsunami problem solved using Thetis'
    ylab1 += '$J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
    tol = 1.5e10      # Note J ~ 2.4e13
else:
    raise ValueError("Problem not recognised.")

# Bootstrap
(Js, nEls, ts) = boot.bootstrap(problem, tol=tol)[1:]

# Plot functional values
plt.title(title)
plt.xlabel(r'\#Elements')
plt.ylabel(ylab1)
if problem in ('firedrake-tsunami', 'thetis-tsunami'):
    plt.semilogx(nEls, Js, basex=10, marker='o', label=ylab1)
else:
    plt.semilogx(nEls, Js, basex=10, marker='o', label=ylab1)
plt.savefig("outdata/bootstrapping/" + problem + ".pdf", bbox_inches='tight')
plt.show()

# Plot timings
plt.clf()
plt.title(title)
plt.xlabel(r'\#Elements')
plt.ylabel(ylab2)
if problem in ('firedrake-tsunami', 'thetis-tsunami'):
    plt.loglog(nEls, ts, basex=10, basey=10, marker='x', label=ylab2)
else:
    plt.loglog(nEls, ts, basex=10, basey=10, marker='x', label=ylab2)
plt.savefig("outdata/bootstrapping/" + problem + "-timings.pdf", bbox_inches='tight')
plt.show()
