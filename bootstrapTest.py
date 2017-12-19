import numpy as np
import matplotlib.pyplot as plt

import utils.bootstrapping as boot


Js = []         # Container for objective functional values
diff = 1        # Initialise 'difference of differences'
tol = 1e-3      # Threshold for convergence
i = 0           # Step counter
maxIter = 10    # Maximum allowed iterations
advDiff = bool(input("\nHit anything except enter to consider advection-diffusion. "))
filename = "advection-diffusion" if advDiff else "shallow-water"

print("\n ***** " + filename + " *****\n")
for i in range(maxIter):
    n = pow(2, i)
    Js.append(boot.solverAD(n) if advDiff else boot.solverSW(n))
    toPrint = "n = %3d, J = %6.4f, " % (n, Js[-1])
    if i > 1:
        diff = np.abs(np.abs(Js[-2] - Js[-3]) - np.abs(Js[-1] - Js[-2]))
        toPrint += "diff : %6.4f" % diff
    print(toPrint)

    if diff < tol:
        print("Converged to J = %.4f in %d iterations" % (Js[-1], i))
        break
    i += 1

plt.gcf()
plt.xlabel(r'Elements per unit in x- and y-directions')
plt.ylabel(r'Objective functional')
plt.semilogx([pow(2, i) for i in range(len(Js))], Js, basex=2)
plt.savefig("outdata/bootstrapping/" + filename + ".pdf", bbox_inches='tight')
plt.show()
