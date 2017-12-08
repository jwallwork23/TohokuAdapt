import utils.options as opt
import utils.timeseries as tim

op = opt.Options(rm=60)
dirName = input("dirName: ") or 'plots/firedrake-tsunami/TohokuXCoarse/'
iEnd = int(input("iEnd: ") or 1500)
useAdjoint = bool(input("useAdjoint?: "))
plot = bool(input("Hit anything except enter to plot an existing timeseries. "))

if not plot:
    name = input("Enter a name for this time series (e.g. 'goalBased8-12-17'): ")

for gauge in ("P02", "P06"):
    if plot:
        tim.plotGauges(gauge, dirName, iEnd, op=op)
    else:
        tim.gaugeTimeseries(gauge, dirName, iEnd, op=op, name=name)
