# Map a behavioral opamp model to a transistor-level circuit netlist based on gmid method,
# which will be used for further fine-tuning/optimization, 
# as well as virtuoso schematic and layout generation.
# Also the simulations of transistor-level circuits can be performed.

import os
import argparse
from netlist_generator import *
from utils import *

work_dir = os.getcwd()

parser = argparse.ArgumentParser("Transistor-level circuit netlist mapping and simulation")
parser.add_argument('--TITLE',  action='store_true',  help='print title')
parser.add_argument('--MAP',  action='store_true',  help='map the circuit netlists')
parser.add_argument('--SIM',  action='store_true',  help='simulate the transistor-level circuit')
parser.add_argument('--SIMALL',  action='store_true',  help='simulate all the transistor-level circuits')
parser.add_argument('--OUT',  action='store_true',  help='generate the netlist out file for the layout negeration')
parser.add_argument('--REOPT1',  action='store_true',  help='re-optimizate the circuit on gm over id target')
parser.add_argument('--REOPT2',  action='store_true',  help='re-optimizate the transistor-level circuit')
parser.add_argument('--Debug',  action='store_true',  help='output the debug information')
parser.add_argument('--opt_iter', type=int, default=100, help='re-optimization iteratons nums')
parser.add_argument('--lb_ratio', type=float, default=0.5, help='the lower bound ratio of optimization sapce')
parser.add_argument('--ub_ratio', type=float, default=1.5, help='the upper bound ratio of optimization sapce')
parser.add_argument('--sample_nums', type=float, default=20, help='the initial sampling nums')
parser.add_argument('--intrinsic_gain_times', type=float, default=1.5, help='the multiple of the intrinsic gain is taken while mapping a single mos in the gmid method')
args = parser.parse_args()

if args.TITLE:
    print(' /////////////////////////////////////////////')
    print(' ///    TRANSISTOR-LEVEL CIRCUIT MAPPING    ///')
    print(' /////////////////////////////////////////////')
    print(' Map behavioral-level opamp model to transistor-level netlist')
    print(' with gmid method.')
    print(' Testbench generation & simulation / gm-level optimization / size-level optimization')
    print(' can also be performed.')
    print(' Version 1.0.\n')

ig_time = args.intrinsic_gain_times

if args.Debug:
    Debug = True
    print('[Debug] using debug mod...')
    print('[Debug] the intrinsic gain times: {}'.format(ig_time))
else:
    Debug = False
    print('[Debug] using non-debug mod...')
behavioral_model_path = './behv_circuits/'

if args.SIMALL:
    all_circuits = os.listdir(r'./behv_circuits')
    all_good_circuits = []
    print('[Info] Simulating all {} circuits that found under path: {}'.format(len(all_circuits), behavioral_model_path))

"""input: the results optimizted by date22's method"""
# parse the param file and read the value and ports of each gm/gds/r/c
selected_circuits_list = ['4ygYTM']
#selected_circuits_list = ['4ygYTM', 'CMJRbs', 'Uf8gGW']
if args.SIMALL:
    selected_circuits_list = all_circuits

circuit_data_dict = {}
device_features = ['type','value', 'unit', 'left_port', 'right_port', 'parasitic', 'circuit_name']

for circuit in selected_circuits_list:
    param_path = behavioral_model_path+circuit+'/circuit/param'
    netlist_path = behavioral_model_path+circuit+'/circuit/netlist/netlist'
    circuit_dict = parse_behavioral_model(circuit, device_features, param_path, netlist_path)
    circuit_data = pd.DataFrame(circuit_dict)
    inst_list = circuit_data.columns.values.tolist()
    # for inst in inst_list:
    #     print(inst+': '+str(circuit_data.loc['value'][inst])+' '+circuit_data.loc['unit'][inst])
    circuit_data_dict.update({circuit:circuit_data})

os.chdir(work_dir)

"""mapping all selected circuits and generate netlists"""
map_circuits_list = ['4ygYTM']
#selected_circuits_list = ['4ygYTM', 'CMJRbs', 'Uf8gGW']
if args.SIMALL:
    map_circuits_list = all_circuits
if args.MAP:
    for circuit in map_circuits_list:
        circuit_data = circuit_data_dict[circuit]
        print('[Info] performing transistor circuit netlist mapping of {}...'.format(circuit))
        os.chdir('./mapped_circuits/')
        os.system('rm -rf ./{}'.format(circuit))
        os.system('mkdir {}'.format(circuit))
        os.chdir('./{}'.format(circuit))
        generate_netlist(ig_time, circuit_data, Debug)
        os.chdir(work_dir)
    print('[Info] all behavioral models are mapping to schematic netlists!')

"""run transistor-level simulation"""
sim_circuits_list = ['4ygYTM']
#selected_circuits_list = ['4ygYTM', 'CMJRbs', 'Uf8gGW']
if args.SIMALL:
    sim_circuits_list = all_circuits

satisfied_circuits = []
if args.SIM:
    for circuit in sim_circuits_list:
        print('[Info] performing transistor circuit simulation of {}...'.format(circuit))
        os.chdir('./mapped_circuits/')
        os.chdir('./{}'.format(circuit))
        gain_init, pm_init, gbw_init, power_init = sim_only(circuit)
        # print('[Info] the initial simulation results are:')
        # print('       gain={}, pm={}, gbw={}, power={}'.format(gain_init, pm_init, gbw_init, power_init))
        res_init = sim_and_getRes(circuit, gain_init, pm_init, gbw_init, power_init)
        os.chdir(work_dir)
        constr0_current = res_init[0][1]
        constr1_current = res_init[0][2]
        constr2_current = res_init[0][3]
        constr3_current = res_init[0][4]
        if constr0_current <=0 and constr1_current<=0 and constr2_current<=0 and constr3_current<=0:
            print('[Suecss] the initial mapping circuit is satisfied with the constraints!')
            satisfied_circuits.append(circuit)
            #all_good_circuits.append(circuit)
        else:
            print('[Info] the initial mapping circuit is not satisfied with the constraints...')
            reopt = True
    print('[Info] all circuit simulations finished!')
    print('[Info] which satisfies the initial constraints are: {}'.format(satisfied_circuits))

"""generate the netlist out file for all the selected circuits"""
out_circuits_list = ['4ygYTM']
#selected_circuits_list = ['4ygYTM', 'CMJRbs', 'Uf8gGW']
circuit_param_dict = {}
circuit_gmidTarget_dict = {}
if args.SIMALL:
    out_circuits_list = all_circuits
if args.OUT:
    for circuit in out_circuits_list:
        circuit_data = circuit_data_dict[circuit]
        print('[Info] genarating the netlist out file of {}...'.format(circuit))
        os.chdir('./mapped_circuits/')
        os.chdir('./{}'.format(circuit))
        all_param_dict, gmid_target_dict = generate_netlistOut(ig_time, circuit_data)
        circuit_param_dict.update({circuit:all_param_dict})
        circuit_gmidTarget_dict.update({circuit:gmid_target_dict})
        #print(all_param_dict)
        os.chdir(work_dir)
    print('[Info] all the netlist out files generation finished!')

"""re-optimizate the circuit on gm over id target"""
reOpt1_circuits_list = ['4ygYTM']
#selected_circuits_list = ['4ygYTM', 'CMJRbs', 'Uf8gGW']
if args.SIMALL:
    reOpt1_circuits_list = all_circuits
opt_satisfied_circuits = []
if args.REOPT1:
    for circuit in reOpt1_circuits_list:
        update = 0
        circuit_data = circuit_data_dict[circuit]
        print('[Info] re-optimizating the circuit {} on gmOverId...'.format(circuit))
        init_gmid_target = circuit_gmidTarget_dict[circuit]
        opt_space = set_gmidTarget_bound(args.lb_ratio, args.ub_ratio, init_gmid_target, debug=Debug)

        # optimization begain
        best_x = opt_space.sample(1)
        #best_y = function_wrapper_gmid(work_dir, circuit, best_x, circuit_data, gain_init, pm_init, gbw_init, power_init, Debug)
        best_y = [[0,85,55,0.7,-250]]
        reOpt = GeneralBO(opt_space, num_obj=1, num_constr=4, rand_sample=args.sample_nums)
        for i in range(args.opt_iter):
            x_new = reOpt.suggest(n_suggestions=1)
            y_new = function_wrapper_gmid(ig_time, work_dir, circuit, x_new, circuit_data, gain_init, pm_init, gbw_init, power_init, Debug)
            if Debug:
                print('[Debug] x_new: {}'.format(x_new))
                print('[Debug] y_new: {}'.format(y_new))
            reOpt.observe(x_new, y_new)
            best_x, best_y, update_tmp = get_best(best_x, best_y, x_new, y_new)
            update += update_tmp
            if Debug:
                print('[Info] current best_x: {}'.format(best_x))
            print('[Info] iter {}: best_foms={}, best_gain={}, best_pm={}, best_gbw={}, best_power={}'.format(i+1, -1*best_y[0][0], 85-best_y[0][1], 55-best_y[0][2], 0.7-best_y[0][3], best_y[0][4]+250))
            best_res = str(-1*best_y[0][0]) + ' ' + str(85-best_y[0][1]) + ' ' + str(55-best_y[0][2]) + ' ' + str(0.7-best_y[0][3]) + ' ' + str(best_y[0][4]+250) + '\n'
            os.system('echo \'{}\' >> best_preSim_res.txt'.format(best_res))

        if update >= 1:
            print('[Info] gmOverId {} re-optimization success!'.format(circuit))
            print('[Info] re-genarating the netlist out file of {}...'.format(circuit))
            param_dict = generate_netlistOut_opt_gmid(ig_time, circuit_data, best_x)
            opt_satisfied_circuits.append(circuit)
            if args.SIMALL:
                all_good_circuits.append(circuit)
        else:
            print('[Info] gmOverId {} re-optimization fail...'.format(circuit))
        os.chdir(work_dir)
    print('[Info] all circuits\' gmOverId re-optimization finished!')
    print('[Info] which satisfies the initial constraints are: {}'.format(opt_satisfied_circuits))

if args.SIMALL:
    if len(all_good_circuits)>0:
        print('[Info] find {} circuits that can satisfied with the constraints'.format(len(all_good_circuits)))
        print(all_good_circuits)
    else:
        print('no circuit can satisfied with the constraints...')
