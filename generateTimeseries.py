import utils.options as opt
import utils.timeseries as tim

op = opt.Options(rm=60)
dirName = 'plots/firedrake-tsunami/TohokuXCoarse/'
iEnd = 1500
plot = bool(input("Hit anything except enter to plot an existing timeseries. "))

if not plot:
    adaptive = bool(input("Hit anything except enter if using adaptivity. "))
    name = input("Enter a name for this time series (e.g. 'goalBased8-12-17'): ") or 'test'

for gauge in ("P02", "P06"):
    if plot:
        tim.plotGauges(gauge, dirName, iEnd, op=op)
    else:
        tim.gaugeTimeseries(gauge, dirName, iEnd, op=op, name=name, adaptive=adaptive)
