import matplotlib.pyplot as plt

import utils.bootstrapping as boot


advDiff = bool(input("\nHit anything except enter to consider advection-diffusion. "))
filename = "advection-diffusion" if advDiff else "shallow-water"
print("\n ***** " + filename + " *****\n")
(Js, nEls) = boot.bootstrap(advDiff)[1:]

plt.gcf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ylab = r'Objective functional'
if advDiff:
    plt.title(r'Advection-diffusion test case on $[0,4]\times[0,1]$')
    ylab += ' $J(\phi)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\phi\:\mathrm{d}x\mathrm{d}t$'
    plt.xlabel(r'\#Elements')
else:
    plt.title(r'Non-rotating shallow water test case on $[0,2]\times[0,2]$')
    ylab += ' $J(u,v,\eta)=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int_A\eta\:\mathrm{d}x\mathrm{d}t$'
    plt.xlabel(r'\#Elements')
plt.ylabel(ylab)
# plt.semilogx([pow(2, i) for i in range(len(Js))], Js, basex=2)
plt.semilogx(nEls, Js, basex=10)
plt.savefig("outdata/bootstrapping/" + filename + ".pdf", bbox_inches='tight')
plt.show()
