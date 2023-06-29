from pydoc import stripid
import re
import pandas as pd
import numpy as np
import os

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.general import GeneralBO
from pymoo.util.dominator import Dominator

from netlist_generator import get_time, generate_netlist_opt, generate_netlist_opt_gmid, generate_simScript, wmax, wmin

""" about the unit """
# 1m = 1e-3, 1u = 1e-6, 1n = 1e-9, 1p = 1e-12, 1f = 1e-15
# 1M = 1e6,  1G = 1e9
# gm:S, gds:S, id:A, vgs:V
# W/L after calculation: m (meter)

""" useful functions for mapping and simulation """
def parse_param(param_path):
    param_dict = {}
    param_file = open(param_path, 'r')
    param_lines = param_file.readlines()
    for line in param_lines:
        line = line.strip()
        match = re.search(r'^\S+\s+(\S+)\s+\S+\s+(\S+)$', line)
        device_name = match.groups()[0]
        device_value = match.groups()[1]
        param_dict.update({device_name:float(device_value)})
    return param_dict

def macth_device(netlist_line):
    parasitic = 0
    line = netlist_line.strip()
    match1 = re.search(r'^(G_\S+)\s+[(](\S+)\s+0\s+(\S+)\s+0[)]\s+(\S+)\s+(\S+)$', line)
    match2 = re.search(r'^([R,C]_\S+)\s+[(](\S+)\s+(\S+)[)]\s+(\S+)\s+(\S+)$', line)
    if match1:
        device_name = match1.groups()[0]
        left_port = match1.groups()[2]  # pay attention to the port direction of vccs here
        right_port = match1.groups()[1]
        device_type = match1.groups()[3]
        device_exp = match1.groups()[4]
    elif match2:
        device_name = match2.groups()[0]
        left_port = match2.groups()[1]
        right_port = match2.groups()[2]
        device_type = match2.groups()[3]
        device_exp = match2.groups()[4]
    else:
        print('[Error] can not match any device !')
    
    match_prs1 = re.search(r'^\S+prs$', device_name)
    match_prs2 = re.search(r'^\S+_0\d', device_name)
    match_prs3 = re.search(r'^\S+_L\d', device_name)
    if match_prs1 or match_prs2 or match_prs3:
        parasitic = 1

    return device_name, device_type, device_exp, left_port, right_port, parasitic

def cal_device_value(dev_exp, param_dic):
    match1  = re.search(r'^\S+=(\S+)[/](\S+)[*](\S+)$', dev_exp)
    match11 = re.search(r'^\S+=(\S+)[/](\d.\d+)[*](\S+)$', dev_exp)
    match2  = re.search(r'^\S+=(\S+)[*](\S+)$', dev_exp)
    match22 = re.search(r'^\S+=(\S+)([M,n])$', dev_exp)
    # match11 must be matched before match1
    if match11:
        value = param_dic[match1.groups()[0]] / float(match1.groups()[1])
        times_unit = match1.groups()[2]
    elif match1:
        value = param_dic[match1.groups()[0]] / param_dic[match1.groups()[1]]
        times_unit = match1.groups()[2]
    elif match2:
        value = param_dic[match2.groups()[0]]
        times_unit = match2.groups()[1]
    elif match22:
        value = float(match22.groups()[0])
        times_unit = match22.groups()[1]
    else:
        print('[Error] device value parse failed1 !')
    
    match3 = re.search(r'^-(\d)([M,K,m,p,n])$', times_unit)
    match4 = re.search(r'^(\d)([M,K,m,p,n])$', times_unit)
    match5 = re.search(r'^([M,n])$', times_unit)
    if match3 :
        value = value*float(match3.groups()[0])*-1
        unit  = match3.groups()[1]
    elif match4 :
        value = value*float(match4.groups()[0])
        unit  = match4.groups()[1]
    elif match5:
        value = value
        unit  = match5.groups()[0]
    else:
        print('[Error] device value parse failed2 !')
    
    return round(value, 3), unit

def parse_behavioral_model(circuit, device_features, param_path, netlist_path):
    circuit_dict = {}
    param_dict = parse_param(param_path)
    netlist_file = open(netlist_path, 'r')
    netlist_lines = netlist_file.readlines()
    for netlist_line in netlist_lines[1:]: # not include vsource
        device_name, device_type, device_exp, left_port, right_port, parasitic = macth_device(netlist_line)
        device_value, device_unit = cal_device_value(device_exp, param_dict)
        # # limit the size of non-parasitic resistors and capacitors
        # if parasitic!=1 and device_type=='capacitor':
        #     if device_value > 0.5:
        #         device_value = 0.5
        #     elif device_value < 0.03:
        #         device_value = 0.03
        #     else:
        #         pass
        # elif parasitic!=1 and device_type=='resistor':
        #     if device_value > 9.8:
        #         device_value = 10.4551
        #     else:
        #         pass
        # potential loss of performance
        # re-optimizing RC may make more sense
        # or modify the bound of RC during behavioral level optimization
        dev_features = [device_type, device_value, device_unit, left_port, right_port, parasitic, circuit]
        circuit_dict.update({device_name:pd.Series(dev_features, index=device_features)})
    return circuit_dict

def sim_only(circuit_name):
    # clear logs
    os.system('rm -rf err sim.* psf sim_results .cadence')
    os.system('rm -rf ~/CDS.log*')
    os.chdir('netlist')
    os.system('rm -rf artSimEnvLog input.scs runSimulation')
    os.chdir('..')
    print('[Info] simulating the circuit {} with spectre...'.format(circuit_name))
    # run simulation
    os.system('ocean -replay ./sim_opamp.ocn -log sim.log > err 2>&1')
    
    sim = os.path.exists('./sim.out')
    if sim== True:
        out       = open('./sim.out','r')
        lines     = out.readlines()
        stripped1 = lines[2].strip()
        match1    = re.search(r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$', stripped1)
        if match1:
            print('[Info] simulation sucessed!')
            gain  = float(match1.groups()[0])
            pm    = float(match1.groups()[1])
            gbw   = float(match1.groups()[2])/1e6
            power = float(match1.groups()[3])*1e6
            out.close()
            print('[Info] gain={}db pm={}deg gbw={}MHz power={}uW'.format(gain, pm, gbw, power))
            return gain, pm, gbw, power
        else:
            print('[Error] incomplete simulation!')
            return 0.1, 0.1, 0.1, 2000
    else:
        print('[Error] simulation failed!')
        return 0.1, 0.1, 0.1, 2000

def sim_and_getRes(circuit_name, init_gain, init_pm, init_gbw, init_power):
    # clear logs
    os.system('rm -rf err sim.* psf sim_results .cadence')
    os.system('rm -rf ~/CDS.log*')
    os.chdir('netlist')
    os.system('rm -rf artSimEnvLog input.scs runSimulation')
    os.chdir('..')
    print('[Info] simulating the circuit {} with spectre...'.format(circuit_name))
    # run simulation
    os.system('ocean -replay ./sim_opamp.ocn -log sim.log > err 2>&1')

    # get simulation results
    # set the optimization goal and constraints here
    # here we should sacrifice gain for margin

    '''   prb_constr0     gain      >= 85  dB    '''
    '''   prb_constr1     pm        >= 55        '''
    '''   prb_constr2     gbw       >= 0.7 MHz   '''
    '''   prb_constr3     power     <= 250 uW    '''
    '''   goal            fom_s     maximize     '''

    gain_target  = 85
    pm_target    = 55
    gbw_target   = 0.7
    power_target = 250

    sim = os.path.exists('./sim.out')
    if sim== True:
        out       = open('./sim.out','r')
        lines     = out.readlines()
        stripped1 = lines[2].strip()
        match1    = re.search(r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$', stripped1)
        if match1:
            print('[Info] simulation sucessed!')
            gain  = float(match1.groups()[0])
            pm    = float(match1.groups()[1])
            gbw   = float(match1.groups()[2])/1e6
            power = float(match1.groups()[3])*1e6
            out.close()

            prb_constr0 = -1*gain+gain_target
            prb_constr1 = -1*pm+pm_target
            prb_constr2 = -1*(gbw)+gbw_target
            prb_constr3 = power - power_target
            fom_s       = gbw*(10*1e3) / (power*1e-3)
            goal        = -fom_s
            print('[Info] gain={}db pm={}deg gbw={}MHz power={}uW'.format(gain, pm, gbw, power))
            print('[Info] FOMs={}'.format(fom_s))
        else:
            print('[Error] incomplete simulation!')
            goal        = -0.1 #(np.random.rand()+1)*100
            prb_constr0 = 0.1 #(np.random.rand()+1)*-10
            prb_constr1 = 0.1 #(np.random.rand()+1)*-10
            prb_constr2 = 0.1 #(np.random.rand()+1)*0.2
            prb_constr3 = 50 #(np.random.rand()+1)*50
    else:
        print('[Error] simulation failed!')
        goal        = -0.1 #(np.random.rand()+1)*100
        prb_constr0 = 0.1 #(np.random.rand()+1)*-10
        prb_constr1 = 0.1 #(np.random.rand()+1)*-10
        prb_constr2 = 0.1 #(np.random.rand()+1)*0.2
        prb_constr3 = 50 #(np.random.rand()+1)*50

    o = [goal]
    o = np.array(o).reshape(1,1)
    c = []
    c.append(prb_constr0)
    c.append(prb_constr1)
    c.append(prb_constr2)
    c.append(prb_constr3)
    c = np.array(c).reshape(1,4)

    res = np.hstack([o,c])

    return res

def preprocessing_netlist(circuit_name, debug):
    postLayout_netlist = open('./opamp_'+circuit_name+'.pex.netlist', 'r')
    print('[Info] performing netlist preprocessing...')
    processed_netlist = open('./post_netlist', 'w')
    lines = postLayout_netlist.readlines()
    for line in lines:
        stripped = line.strip()
        match = re.search(r'^ends\s+(\S+)$', stripped)
        if match:
            wrong_name = match.groups()[0]
            right_name = "opamp_"+circuit_name
            line = re.sub(wrong_name, right_name, line)
            if debug:
                print('[Debug] find wrong model name \'{}\' in netlist'.format(wrong_name))
                print('[Debug] replaced with \'{}\''.format(right_name))
            processed_netlist.writelines(line)
        else:
            processed_netlist.writelines(line)
    processed_netlist.close()
    postLayout_netlist.close()

def postLayout_sim(circuit_name, parasitic_path, debug=False):
    # setup
    os.system('rm -rf postLayout_sim')
    os.system('mkdir postLayout_sim')
    os.chdir('postLayout_sim')
    generate_simScript()
    os.system('mkdir netlist')
    os.chdir('./netlist')
    # copy post layout parasitic file
    os.system('cp {}/OPAMP_{}.pex.netlist* ./'.format(parasitic_path, circuit_name.upper()))
    #preprocessing_netlist(circuit_name, debug)
    # write header
    netlist_header = open('./netlistHeader', 'w')
    netlist_header.writelines('// Automated generated for post layout simulation\n')
    netlist_header.writelines('// Generated for: spectre\n')
    local_time = get_time()
    netlist_header.writelines('// Generated on: {}\n'.format(local_time))
    netlist_header.writelines('simulator lang=spectre\n')
    netlist_header.writelines('global vdd gnd\n')
    netlist_header.writelines('// Header End \n')
    netlist_header.close()
    # write footer
    netlist_footer = open('./netlistFooter', 'w')
    netlist_footer.writelines('// Footer End')
    netlist_footer.close()
    # write netlist
    netlist = open('./netlist', 'w')
    netlist.writelines('include "OPAMP_{}.pex.netlist"\n'.format(circuit_name.upper()))
    #netlist.writelines('include "post_netlist"\n'.format(circuit_name))
    netlist.writelines('ID0 (vdd net_vb_n) isource dc=1u type=dc \n')
    netlist.writelines('ID1 (net_vb_p gnd) isource dc=1u type=dc \n')
    netlist.writelines('V0 (net_vip gnd) vsource dc=900.0m mag=1 type=sine\n')
    netlist.writelines('V1 (vdd gnd) vsource dc=1.8 type=dc\n')
    netlist.writelines('C0 (net_vin gnd) capacitor c=1F \n')
    netlist.writelines('L0 (net_vin net_vo) inductor l=1G \n')
    netlist.writelines('C1 (net_vo gnd) capacitor c=10.0n \n')
    netlist.writelines('X0 (gnd vdd net_vo net_vb_n net_vip net_vin net_vb_p) OPAMP_{}\n'.format(circuit_name.upper()))
    netlist.close()
    #os.system('chmod u+x *')
    os.chdir('..')

    # clear logs
    os.system('rm -rf ~/CDS.log*')
    # run simulation
    print('[Info] simulating the circuit {} with spectre...'.format(circuit_name))
    os.system('ocean -replay ./sim_opamp.ocn -log sim.log > err 2>&1')
    
    sim = os.path.exists('./sim.out')
    if sim== True:
        out       = open('./sim.out','r')
        lines     = out.readlines()
        stripped1 = lines[2].strip()
        match1    = re.search(r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$', stripped1)
        if match1:
            print('[Info] simulation sucessed!')
            gain  = float(match1.groups()[0])
            pm    = float(match1.groups()[1])
            gbw   = float(match1.groups()[2])/1e6
            power = float(match1.groups()[3])*1e6
            out.close()
            print('[Info] gain={}db pm={}deg gbw={}MHz power={}uW'.format(gain, pm, gbw, power))
            os.chdir('..')
            return gain, pm, gbw, power
        else:
            print('[Error] incomplete simulation!')
            os.chdir('..')
            return 0.1, 0.1, 0.1, 2000
    else:
        print('[Error] simulation failed!')
        os.chdir('..')
        return 0.1, 0.1, 0.1, 2000


""" circuit re-optimization after mapping to transistor-level """
def set_param_bound(lb_ratio, ub_ratio, param_dict):
    # tuning the W of specific mos only
    space_list = []
    for key in param_dict.keys():
        initial_value = param_dict[key]
        lb = initial_value*lb_ratio
        if lb < wmin:
            lb = wmin
        ub = initial_value*ub_ratio
        space_list.append({'name':key, 'type':'num', 'lb':lb, 'ub':ub})
    opt_space = DesignSpace().parse(space_list)
    return opt_space

def set_gmidTarget_bound(lb_ratio, ub_ratio, gmidTarget_dict, debug=None):
    if debug==None:
        debug = False
    # tuning the W of specific mos only
    space_list = []
    for key in gmidTarget_dict.keys():
        initial_value = gmidTarget_dict[key]
        lb = 5 #initial_value*lb_ratio
        ub = 15 #initial_value*ub_ratio
        space_list.append({'name':key, 'type':'num', 'lb':lb, 'ub':ub})
    opt_space = DesignSpace().parse(space_list)
    if debug:
        var_names = opt_space.numeric_names
        print('[Debug] {} gm/id parameters to be optimized:'.format(len(var_names)))
        print('[Debug]    {}'.format(var_names))
    return opt_space

def function_wrapper(work_dir, circuit_name, x, circuit_data, init_gain, init_pm, init_gbw, init_power):
    os.chdir(work_dir)
    os.chdir('./mapped_circuits/{}'.format(circuit_name))
    generate_netlist_opt(circuit_data, x)
    res = sim_and_getRes(circuit_name, init_gain, init_pm, init_gbw, init_power)
    return res

def function_wrapper_gmid(intrinsic_gain_times, work_dir, circuit_name, x_gmid, circuit_data, init_gain, init_pm, init_gbw, init_power, debug):
    os.chdir(work_dir)
    os.chdir('./mapped_circuits/{}'.format(circuit_name))
    generate_netlist_opt_gmid(intrinsic_gain_times, circuit_data, x_gmid, debug)
    res = sim_and_getRes(circuit_name, init_gain, init_pm, init_gbw, init_power)
    return res

def extract_pf(points : np.ndarray) -> np.ndarray:
    dom_matrix = Dominator().calc_domination_matrix(points,None)
    is_optimal = (dom_matrix >= 0).all(axis = 1)
    return points[is_optimal]

def get_best(current_best_x, current_best_y, x_new, y_new):
    update = 0

    goal_current = -1*current_best_y[0][0]
    constr0_current = current_best_y[0][1]
    constr1_current = current_best_y[0][2]
    constr2_current = current_best_y[0][3]
    constr3_current = current_best_y[0][4]

    goal_new = -1*y_new[0][0]
    constr0_new = y_new[0][1]
    constr1_new = y_new[0][2]
    constr2_new = y_new[0][3]
    constr3_new = y_new[0][4]

    if constr0_current <=0 and constr1_current<=0 and constr2_current<=0 and constr3_current<=0:
        if constr0_new <=0 and constr1_new<=0 and constr2_new<=0 and constr3_new<=0:
            if goal_new >= goal_current:
                best_x = x_new
                best_y = y_new
                update = 1
            else:
                best_x = current_best_x
                best_y = current_best_y
                update = 0
        else:
            best_x = current_best_x
            best_y = current_best_y
            update = 0
    else:
        if constr0_new <=0 and constr1_new<=0 and constr2_new<=0 and constr3_new<=0:
            best_x = x_new
            best_y = y_new
            update = 1
        else:
            best_x = current_best_x
            best_y = current_best_y
            update = 0

    return best_x, best_y, update

