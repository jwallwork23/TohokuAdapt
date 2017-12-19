import numpy as np
import matplotlib.pyplot as plt

import utils.bootstrapping as boot


Js = []         # Container for objective functional values
nEls = []       # Container for element counts
diff = 1        # Initialise 'difference of differences'
tol = 1e-3      # Threshold for convergence
i = 0           # Step counter
maxIter = 8     # Maximum allowed iterations
advDiff = bool(input("\nHit anything except enter to consider advection-diffusion. "))
filename = "advection-diffusion" if advDiff else "shallow-water"

print("\n ***** " + filename + " *****\n")
for i in range(maxIter):
    n = pow(2, i)
    J, nEle = boot.solverAD(n) if advDiff else boot.solverSW(n)
    Js.append(J)
    nEls.append(nEle)
    toPrint = "n = %3d, nEle = %6d, J = %6.4f, " % (n, nEle, Js[-1])
    if i > 1:
        diff = np.abs(np.abs(Js[-2] - Js[-3]) - np.abs(Js[-1] - Js[-2]))
        toPrint += "diff : %6.4f" % diff
    print(toPrint)

    if diff < tol:
        print("Converged to J = %.4f in %d iterations" % (Js[-1], i))
        break
    i += 1

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
