import utils.options as opt
import utils.timeseries as tim

op = opt.Options(rm=60)

dirName = input("dirName: ") or 'plots/firedrake-tsunami/TohokuXCoarse/'
iEnd = int(input("iEnd: ") or 1500)
useAdjoint = bool(input("useAdjoint?: "))

for gauge in ("P02", "P06"):
    tim.gaugeTimeseries(gauge, dirName, iEnd, op=op)
