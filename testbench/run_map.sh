# set env
source /apps/EDAs/ic615.cshrc

# perform mapping or simulation or netlist out generation or re-optimization
python -W ignore main.py --TITLE --MAP --SIM  --SIMALL --OUT --REOPT1 --intrinsic_gain_times 1
#python -W ignore main.py --TITLE --MAP --SIM --OUT --REOPT1 --intrinsic_gain_times 1
