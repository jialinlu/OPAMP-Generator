# generate transistor-level circuit netlists with parsered behavioral opamp model

import os
import time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from gm_lib import *
import pandas as pd

"""global variables"""
lmin = 1.8e-7      # tsmcrf018
lmax = 5.001e-7    # tsmcrf018
wmin = 1.5e-06     # tsmcrf018
wmax = 8.001e-06   # tsmcrf018
node_vin = 'net_vin'
node_vo  = 'net_vo'
gnd      = '0'
model_sections = ['tt','tt_3v','tt_na','tt_3vna','tt_m','tt_3m','tt_bip','tt_bip3','dio','dio3','dio_dnw','tt_res','tt_mim','tt_rfmos','tt_rfmos33','tt_rfmim','tt_rfind','tt_rfmvar','tt_rfjvar','tt_rfres_sa','tt_rfres_rpo','tt_rfres_hri','tt_rfesd','tt_bbmvar']
model_path = "/apps/share/gm_id_mapping/tsmc18rf/1p6m/models/spectre/rf018.scs"

""" build gmid LUT and calculate W/L """
# raw data path

pwd_path = os.getcwd()

nmos_lut_path = pwd_path + '/tsmc18rf_1p6m_lut/nmos2v/200steps/'
pmos_lut_path = pwd_path + '/tsmc18rf_1p6m_lut/pmos2v/200steps/'

# L is 200 points sampled from [0.18u, 2u], because the PDK used is tsmc18rf
L_list = [1.8e-07, 1.89145728643216e-07, 1.98291457286432e-07, 2.07437185929648e-07, 2.16582914572864e-07, 2.2572864321608e-07, 2.34874371859296e-07, 2.44020100502512e-07, 2.53165829145729e-07, 2.62311557788945e-07, 2.71457286432161e-07, 2.80603015075377e-07, 2.89748743718593e-07, 2.98894472361809e-07, 3.08040201005025e-07, 3.17185929648241e-07, 3.26331658291457e-07, 3.35477386934673e-07, 3.44623115577889e-07, 3.53768844221106e-07, 3.62914572864322e-07, 3.72060301507538e-07, 3.81206030150754e-07, 3.9035175879397e-07, 3.99497487437186e-07, 4.08643216080402e-07, 4.17788944723618e-07, 4.26934673366834e-07, 4.3608040201005e-07, 4.45226130653266e-07, 4.54371859296482e-07, 4.63517587939699e-07, 4.72663316582915e-07, 4.81809045226131e-07, 4.90954773869347e-07, 5.00100502512563e-07, 5.09246231155779e-07, 5.18391959798995e-07, 5.27537688442211e-07, 5.36683417085427e-07, 5.45829145728643e-07, 5.54974874371859e-07, 5.64120603015075e-07, 5.73266331658291e-07, 5.82412060301508e-07, 5.91557788944724e-07, 6.0070351758794e-07, 6.09849246231156e-07, 6.18994974874372e-07, 6.28140703517588e-07, 6.37286432160804e-07, 6.4643216080402e-07, 6.55577889447236e-07, 6.64723618090452e-07, 6.73869346733668e-07, 6.83015075376884e-07, 6.92160804020101e-07, 7.01306532663317e-07, 7.10452261306533e-07, 7.19597989949749e-07, 7.28743718592965e-07, 7.37889447236181e-07, 7.47035175879397e-07, 7.56180904522613e-07, 7.65326633165829e-07, 7.74472361809045e-07, 7.83618090452261e-07, 7.92763819095477e-07, 8.01909547738694e-07, 8.1105527638191e-07, 8.20201005025126e-07, 8.29346733668342e-07, 8.38492462311558e-07, 8.47638190954774e-07, 8.5678391959799e-07, 8.65929648241206e-07, 8.75075376884422e-07, 8.84221105527638e-07, 8.93366834170854e-07, 9.0251256281407e-07, 9.11658291457287e-07, 9.20804020100503e-07, 9.29949748743719e-07, 9.39095477386935e-07, 9.48241206030151e-07, 9.57386934673367e-07, 9.66532663316583e-07, 9.75678391959799e-07, 9.84824120603015e-07, 9.93969849246231e-07, 1.00311557788945e-06, 1.01226130653266e-06, 1.02140703517588e-06, 1.0305527638191e-06, 1.03969849246231e-06, 1.04884422110553e-06, 1.05798994974874e-06, 1.06713567839196e-06, 1.07628140703517e-06, 1.08542713567839e-06, 1.09457286432161e-06, 1.10371859296482e-06, 1.11286432160804e-06, 1.12201005025125e-06, 1.13115577889447e-06, 1.14030150753769e-06, 1.1494472361809e-06, 1.15859296482412e-06, 1.16773869346733e-06, 1.17688442211055e-06, 1.18603015075377e-06, 1.19517587939698e-06, 1.2043216080402e-06, 1.21346733668341e-06, 1.22261306532663e-06, 1.23175879396985e-06, 1.24090452261306e-06, 1.25005025125628e-06, 1.25919597989949e-06, 1.26834170854271e-06, 1.27748743718593e-06, 1.28663316582914e-06, 1.29577889447236e-06, 1.30492462311557e-06, 1.31407035175879e-06, 1.32321608040201e-06, 1.33236180904522e-06, 1.34150753768844e-06, 1.35065326633165e-06, 1.35979899497487e-06, 1.36894472361809e-06, 1.3780904522613e-06, 1.38723618090452e-06, 1.39638190954773e-06, 1.40552763819095e-06, 1.41467336683417e-06, 1.42381909547738e-06, 1.4329648241206e-06, 1.44211055276381e-06, 1.45125628140703e-06, 1.46040201005025e-06, 1.46954773869346e-06, 1.47869346733668e-06, 1.48783919597989e-06, 1.49698492462311e-06, 1.50613065326632e-06, 1.51527638190954e-06, 1.52442211055276e-06, 1.53356783919597e-06, 1.54271356783919e-06, 1.55185929648241e-06, 1.56100502512562e-06, 1.57015075376884e-06, 1.57929648241205e-06, 1.58844221105527e-06, 1.59758793969848e-06, 1.6067336683417e-06, 1.61587939698492e-06, 1.62502512562813e-06, 1.63417085427135e-06, 1.64331658291456e-06, 1.65246231155778e-06, 1.661608040201e-06, 1.67075376884421e-06, 1.67989949748743e-06, 1.68904522613065e-06, 1.69819095477386e-06, 1.70733668341708e-06, 1.71648241206029e-06, 1.72562814070351e-06, 1.73477386934672e-06, 1.74391959798994e-06,1.75306532663316e-06, 1.76221105527637e-06, 1.77135678391959e-06, 1.78050251256281e-06, 1.78964824120602e-06, 1.79879396984924e-06, 1.80793969849245e-06, 1.81708542713567e-06, 1.82623115577888e-06, 1.8353768844221e-06, 1.84452261306532e-06, 1.85366834170853e-06, 1.86281407035175e-06, 1.87195979899496e-06, 1.88110552763818e-06, 1.8902512562814e-06, 1.89939698492461e-06, 1.90854271356783e-06, 1.91768844221104e-06, 1.92683417085426e-06, 1.93597989949748e-06, 1.94512562814069e-06, 1.95427135678391e-06, 1.96341708542712e-06, 1.97256281407034e-06, 1.98170854271356e-06, 1.99085427135677e-06, 2e-06]

# load data and build LUT
def replace_col(df, new_col_names, value_names):
    new_df = df
    col_names = df.columns.values
    i_col = 0
    i_new_col = 0
    while i_col < len(col_names):
        new_df.rename(columns={col_names[i_col]:str(new_col_names[i_new_col])+value_names[0]}, inplace=True)
        new_df.rename(columns={col_names[i_col+1]:str(new_col_names[i_new_col])+value_names[1]}, inplace=True)
        i_col += 2
        i_new_col += 1
    return new_df

def build_lut(nmos_path, pmos_path):
    # read raw data    
    # nmos
    gm_nmos  = pd.read_csv(nmos_path+'gm.csv')
    gm_nmos  = replace_col(gm_nmos, L_list, ['_vgs', '_gm'])
    gds_nmos = pd.read_csv(nmos_path+'ro.csv') 
    gds_nmos = replace_col(gds_nmos, L_list, ['_vgs', '_gds'])
    id_nmos  = pd.read_csv(nmos_path+'id.csv')
    id_nmos  = replace_col(id_nmos, L_list, ['_vgs', '_id'])
    # pmos
    gm_pmos  = pd.read_csv(pmos_path+'gm.csv')
    gm_pmos  = replace_col(gm_pmos, L_list, ['_vgs', '_gm'])
    gds_pmos = pd.read_csv(pmos_path+'ro.csv')
    gds_pmos = replace_col(gds_pmos, L_list, ['_vgs', '_gds'])
    id_pmos  = pd.read_csv(pmos_path+'id.csv')
    id_pmos  = replace_col(id_pmos, L_list, ['_vgs', '_id'])

    # make column -> L while row(index) -> VGS
    # nmos
    vgs_n = gm_nmos[str(L_list[0])+'_vgs'].values.tolist()
    gm_n_dict  = {}
    gds_n_dict = {}
    id_n_dict  = {}
    for l in L_list:
        gm_n_dict.update({l:pd.Series(gm_nmos[str(l)+'_gm'].values.tolist(), index=vgs_n)}) 
        gds_n_dict.update({l:pd.Series(gds_nmos[str(l)+'_gds'].values.tolist(), index=vgs_n)}) 
        id_n_dict.update({l:pd.Series(id_nmos[str(l)+'_id'].values.tolist(), index=vgs_n)})
    gm_n  = pd.DataFrame(gm_n_dict)
    gds_n = pd.DataFrame(gds_n_dict)
    id_n  = pd.DataFrame(id_n_dict)
    # pmos
    vgs_p = gm_pmos[str(l)+'_vgs'].values.tolist()
    gm_p_dict  = {}
    gds_p_dict = {}
    id_p_dict  = {}
    for l in L_list:
        gm_p_dict.update({l:pd.Series(gm_pmos[str(l)+'_gm'].values.tolist(), index=vgs_p)}) 
        gds_p_dict.update({l:pd.Series(gds_pmos[str(l)+'_gds'].values.tolist(), index=vgs_p)}) 
        id_p_dict.update({l:pd.Series(id_pmos[str(l)+'_id'].values.tolist(), index=vgs_p)})
    gm_p  = pd.DataFrame(gm_p_dict)
    gds_p = pd.DataFrame(gds_p_dict)
    id_p  = pd.DataFrame(id_p_dict)

    return gm_n, gds_n, id_n, gm_p, gds_p, id_p

""" generate netlist/netlist4opt/netlistOut/script4simulation """
def get_time():
    times = time.time()
    local_times = time.localtime(times)
    local_time_asctimes = time.asctime(local_times)
    return local_time_asctimes

def map_unit(unit):
    if unit == 'M':
        times = 1e6
    elif unit == 'K':
        times = 1e3
    elif unit == 'm':
        times = 1e-3
    elif unit == 'p':
        times = 1e-6
    elif unit == 'n':
        times = 1e-9
    else:
        print('[Error] unknown unit !')
    return times

def get_gds(gm_name, circuit_data):
    gds = 0
    gm_right_port = circuit_data.loc['right_port'][gm_name]
    inst_list = circuit_data.columns.values.tolist()
    for inst in inst_list:
        inst_type = circuit_data.loc['type'][inst]
        parasitic = circuit_data.loc['parasitic'][inst]
        left_port = circuit_data.loc['left_port'][inst]
        if inst_type=='resistor' and parasitic==1 and left_port==gm_right_port:
            unit = circuit_data.loc['unit'][inst]
            times = map_unit(unit)
            ro = circuit_data.loc['value'][inst] * times
            gds = 1/ro
            break
    if gds == 0:
        print('[Error] {} get gds failed!'.format(gm_name))
    return gds

def remove_invalid_device(inst_list, circuit_data):
    valid_device_list = []
    invalid_port = []
    for inst in inst_list:
        inst_type = circuit_data.loc['type'][inst]
        left_port = circuit_data.loc['left_port'][inst]
        right_port = circuit_data.loc['right_port'][inst]
        parasitic = circuit_data.loc['parasitic'][inst]
        if inst_type == 'capacitor' or inst_type == 'resistor':
            if parasitic == 0 and left_port == 'net_vin' and (not (right_port in ['net_1', 'net_2'])):
                invalid_port.append(right_port)
    invalid_port.append('net_vin')
    for inst in inst_list:
        inst_type = circuit_data.loc['type'][inst]
        left_port = circuit_data.loc['left_port'][inst]
        right_port = circuit_data.loc['right_port'][inst]
        parasitic = circuit_data.loc['parasitic'][inst]
        if inst_type == 'capacitor' or inst_type == 'resistor':
            if parasitic == 0 and (not (left_port in invalid_port)):
                valid_device_list.append(inst)
        else:
            valid_device_list.append(inst)
    return valid_device_list, invalid_port

def generate_netlist(intrinsic_gain_times, pd_data, debug=None):
    if debug==None:
        debug = False
    # build LUT
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    
    os.system('mkdir netlist')
    os.chdir('./netlist')
    netlist_header = open('./netlistHeader', 'w')
    netlist = open('./netlist', 'w')
    
    # write header
    netlist_header.writelines('// Automated generated for schematic mapping\n')
    netlist_header.writelines('// Generated for: spectre\n')
    local_time = get_time()
    netlist_header.writelines('// Generated on: {}\n'.format(local_time))
    netlist_header.writelines('simulator lang=spectre\n')
    netlist_header.writelines('global 0 vdd! vss!\n')
    netlist_header.writelines('// Header End \n')

    # write gm subckts
    netlist.writelines('// definition of subcircuit\n')
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)
    if debug:
        print('[Debug] Invalid RC port detected: {}'.format(invalid_port))
    gm_index = 0
    gm_dict = {}
    for inst in instance_list:
        if pd_data.loc['type'][inst] == 'vccs':
            unit = pd_data.loc['unit'][inst]
            times = map_unit(unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            left_port = pd_data.loc['left_port'][inst]
            if left_port == node_vin:
                if gm_cal > 0:
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug)
                    gm_name = get_gm_positive_diff(netlist, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
                else:
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_diff(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug)
                    gm_name = get_gm_negative_diff(netlist, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
            else:
                if gm_cal > 0:
                    W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug)
                    gm_name = get_gm_positive_mid(netlist, gm_index, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
                else:
                    W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_mid(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug)
                    gm_name = get_gm_negative_mid(netlist, gm_index, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
            gm_index += 1
        else:
            pass
    netlist.writelines('// End of subcircuit definition \n')
    netlist.writelines('\n')

    # set the differential input
    negative_input = node_vin 
    positive_input = 'net_vip'
    netlist.writelines('V1 (vdd! 0) vsource dc=1.8 type=dc\n')
    netlist.writelines('V0 ({} 0) vsource dc=900.0m mag=1 type=sine\n'.format(positive_input))
    netlist.writelines('C0 (net_vin 0) capacitor c=1F \n')
    netlist.writelines('L0 (net_vin {}) inductor l=1G \n'.format(node_vo))
    netlist.writelines('C1 (net_vo 0) capacitor c=10.0n \n')
    netlist.writelines('ID0 (vdd! net_vb_n) isource dc=1u type=dc\n')
    netlist.writelines('ID1 (net_vb_p 0) isource dc=1u type=dc\n')

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    inst_index      = 0
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst]
        right_port = pd_data.loc['right_port'][inst]
        inst_unit = pd_data.loc['unit'][inst]
        times = map_unit(inst_unit)
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            netlist.writelines('C{} ({} {}) capacitor c={}{}\n'.format(capacitor_index, left_port, right_port, inst_value, inst_unit))
            capacitor_index += 1
        elif inst_type == 'resistor':
            netlist.writelines('R{} ({} {}) resistor r={}{}\n'.format(resistor_index, left_port, right_port, inst_value, inst_unit))
            resistor_index += 1
        elif inst_type == 'vccs':
            inst_name = gm_dict[inst]
            if left_port == node_vin:
                netlist.writelines('I{} (vdd! {} {} {} 0 net_vb_n) {}\n'.format(inst_index, negative_input, positive_input, right_port, inst_name))
            else:
                gm_cal = inst_value*times
                if gm_cal > 0:
                    netlist.writelines('I{} (vdd! {} {} 0 net_vb_n) {}\n'.format(inst_index, left_port, right_port, inst_name))
                else:
                    netlist.writelines('I{} (vdd! {} {} 0 net_vb_p) {}\n'.format(inst_index, left_port, right_port, inst_name))
            inst_index += 1
        else:
            pass
    netlist.writelines('\n')

    netlist_footer = open('./netlistFooter', 'w')
    netlist_footer.writelines('// Footer End')

    netlist_footer.close()
    netlist.close()
    netlist_footer.close()

    os.chdir('..')

    # generate Ocean script for simulation
    generate_simScript()

def generate_netlist_opt(pd_data, param):
    # TODO
    # build LUT
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    
    os.system('rm -rf netlist')
    os.system('mkdir netlist')
    os.chdir('./netlist')
    netlist_header = open('./netlistHeader', 'w')
    netlist = open('./netlist', 'w')
    
    # write header
    netlist_header.writelines('// Automated generated for schematic mapping\n')
    netlist_header.writelines('// Generated for: spectre\n')
    local_time = get_time()
    netlist_header.writelines('// Generated on: {}\n'.format(local_time))
    netlist_header.writelines('simulator lang=spectre\n')
    netlist_header.writelines('global 0 vdd! vss!\n')
    netlist_header.writelines('// Header End \n')

    # write gm subckts
    netlist.writelines('// definition of subcircuit\n')
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)
    gm_index = 0
    gm_dict = {}
    for inst in instance_list:
        if pd_data.loc['type'][inst] == 'vccs':
            unit = pd_data.loc['unit'][inst]
            times = map_unit(unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            left_port = pd_data.loc['left_port'][inst]
            if left_port == node_vin:
                if gm_cal > 0:
                    _, L5, M5, _, L4, M4, _, L3, M3, _, L2, M2, _, L1, M1, _, L0, M0 = cal_positive_diff(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W5 = param['M5_'+str(gm_index)+'_w'][0]
                    W4 = param['M4_'+str(gm_index)+'_w'][0]
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W3 = W2
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    W1 = W0
                    gm_name = get_gm_positive_diff(netlist, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                else:
                    _, L5, M5, _, L4, M4, _, L3, M3, _, L2, M2, _, L1, M1, _, L0, M0 = cal_negative_diff(-1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W5 = param['M5_'+str(gm_index)+'_w'][0]
                    W4 = param['M4_'+str(gm_index)+'_w'][0]
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W3 = W2
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    W1 = W0
                    gm_name = get_gm_negative_diff(netlist, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
            else:
                if gm_cal > 0:
                    _, L2, M2, _, L1, M1, _, L0, M0 = cal_positive_mid(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W1 = param['M1_'+str(gm_index)+'_w'][0]
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    gm_name = get_gm_positive_mid(netlist, gm_index, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                else:
                    _, L4, M4, _, L3, M3, _, L2, M2, _, L1, M1, _, L0, M0 = cal_negative_mid(-1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W4 = param['M4_'+str(gm_index)+'_w'][0]
                    W3 = param['M3_'+str(gm_index)+'_w'][0]
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    W1 = W0
                    gm_name = get_gm_negative_mid(netlist, gm_index, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
            gm_index += 1
        else:
            pass
    netlist.writelines('// End of subcircuit definition \n')
    netlist.writelines('\n')

    # set the differential input
    negative_input = node_vin 
    positive_input = 'net_vip'
    netlist.writelines('V0 ({} 0) vsource dc=900.0m mag=1 type=sine\n'.format(positive_input))
    netlist.writelines('V1 (vdd! 0) vsource dc=1.8 type=dc\n')
    netlist.writelines('C0 (net_vin 0) capacitor c=1F \n')
    netlist.writelines('L0 (net_vin {}) inductor l=1G \n'.format(node_vo))
    netlist.writelines('C1 (net_vo 0) capacitor c=10.0n \n')
    netlist.writelines('ID0 (vdd! net_vb_n) isource dc=1u type=dc\n')
    netlist.writelines('ID1 (net_vb_p 0) isource dc=1u type=dc\n')

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    inst_index      = 0
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst]
        right_port = pd_data.loc['right_port'][inst]
        inst_unit = pd_data.loc['unit'][inst]
        times = map_unit(inst_unit)
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            netlist.writelines('C{} ({} {}) capacitor c={}{}\n'.format(capacitor_index, left_port, right_port, inst_value, inst_unit))
            capacitor_index += 1
        elif inst_type == 'resistor':
            netlist.writelines('R{} ({} {}) resistor r={}{}\n'.format(resistor_index, left_port, right_port, inst_value, inst_unit))
            resistor_index += 1
        elif inst_type == 'vccs':
            inst_name = gm_dict[inst]
            if left_port == node_vin:
                netlist.writelines('I{} (vdd! {} {} {} 0 net_vb_n) {}\n'.format(inst_index, negative_input, positive_input, right_port, inst_name))
            else:
                gm_cal = inst_value*times
                if gm_cal > 0:
                    netlist.writelines('I{} (vdd! {} {} 0 net_vb_n) {}\n'.format(inst_index, left_port, right_port, inst_name))
                else:
                    netlist.writelines('I{} (vdd! {} {} 0 net_vb_p) {}\n'.format(inst_index, left_port, right_port, inst_name))
            inst_index += 1
        else:
            pass
    netlist.writelines('\n')

    netlist_footer = open('./netlistFooter', 'w')
    netlist_footer.writelines('// Footer End')

    netlist_footer.close()
    netlist.close()
    netlist_footer.close()

    os.chdir('..')

    # generate Ocean script for simulation
    generate_simScript()

def generate_netlist_opt_gmid(intrinsic_gain_times, pd_data, param_gmid, debug=None):
    if debug==None:
        debug = False
    
    # build LUT
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    
    os.system('rm -rf netlist')
    os.system('mkdir netlist')
    os.chdir('./netlist')
    netlist_header = open('./netlistHeader', 'w')
    netlist = open('./netlist', 'w')
    
    # write header
    netlist_header.writelines('// Automated generated for schematic mapping\n')
    netlist_header.writelines('// Generated for: spectre\n')
    local_time = get_time()
    netlist_header.writelines('// Generated on: {}\n'.format(local_time))
    netlist_header.writelines('simulator lang=spectre\n')
    netlist_header.writelines('global 0 vdd! vss!\n')
    netlist_header.writelines('// Header End \n')

    # write gm subckts
    netlist.writelines('// definition of subcircuit\n')
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)
    if debug:
        print('[Debug] Invalid RC port detected: {}'.format(invalid_port))
    gm_index = 0
    gm_dict = {}
    for inst in instance_list:
        if pd_data.loc['type'][inst] == 'vccs':
            unit = pd_data.loc['unit'][inst]
            times = map_unit(unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            left_port = pd_data.loc['left_port'][inst]
            if left_port == node_vin:
                if gm_cal > 0:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    if debug:
                        print('[Debug] new gmidTarget_1 and gmidTarget_2 are {}, {}'.format(gmidTarget_1, gmidTarget_2))
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_positive_diff(netlist, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
                else:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    if debug:
                        print('[Debug] new gmidTarget_1 and gmidTarget_2 are {}, {}'.format(gmidTarget_1, gmidTarget_2))
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_diff(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_negative_diff(netlist, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
            else:
                if gm_cal > 0:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    if debug:
                        print('[Debug] new gmidTarget_1 and gmidTarget_2 are {}, {}'.format(gmidTarget_1, gmidTarget_2))
                    W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_positive_mid(netlist, gm_index, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
                else:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    if debug:
                        print('[Debug] new gmidTarget_1 and gmidTarget_2 are {}, {}'.format(gmidTarget_1, gmidTarget_2))
                    W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_mid(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=debug, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_negative_mid(netlist, gm_index, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gm_dict.update({inst:gm_name})
                    if debug:
                        print('[Debug] {} calculation finished, the corresponding gm_cal is {}, the gds_cal is {}'.format(gm_name, gm_cal, gds_cal))
            gm_index += 1
        else:
            pass
    netlist.writelines('// End of subcircuit definition \n')
    netlist.writelines('\n')

    # set the differential input
    negative_input = node_vin 
    positive_input = 'net_vip'
    netlist.writelines('V0 ({} 0) vsource dc=900.0m mag=1 type=sine\n'.format(positive_input))
    netlist.writelines('V1 (vdd! 0) vsource dc=1.8 type=dc\n')
    netlist.writelines('C0 (net_vin 0) capacitor c=1F \n')
    netlist.writelines('L0 (net_vin {}) inductor l=1G \n'.format(node_vo))
    netlist.writelines('C1 (net_vo 0) capacitor c=10.0n \n')
    netlist.writelines('ID0 (vdd! net_vb_n) isource dc=1u type=dc\n')
    netlist.writelines('ID1 (net_vb_p 0) isource dc=1u type=dc\n')

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    inst_index      = 0
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst]
        right_port = pd_data.loc['right_port'][inst]
        inst_unit = pd_data.loc['unit'][inst]
        times = map_unit(inst_unit)
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            netlist.writelines('C{} ({} {}) capacitor c={}{}\n'.format(capacitor_index, left_port, right_port, inst_value, inst_unit))
            capacitor_index += 1
        elif inst_type == 'resistor':
            netlist.writelines('R{} ({} {}) resistor r={}{}\n'.format(resistor_index, left_port, right_port, inst_value, inst_unit))
            resistor_index += 1
        elif inst_type == 'vccs':
            inst_name = gm_dict[inst]
            if left_port == node_vin:
                netlist.writelines('I{} (vdd! {} {} {} 0 net_vb_n) {}\n'.format(inst_index, negative_input, positive_input, right_port, inst_name))
            else:
                gm_cal = inst_value*times
                if gm_cal > 0:
                    netlist.writelines('I{} (vdd! {} {} 0 net_vb_n) {}\n'.format(inst_index, left_port, right_port, inst_name))
                else:
                    netlist.writelines('I{} (vdd! {} {} 0 net_vb_p) {}\n'.format(inst_index, left_port, right_port, inst_name))
            inst_index += 1
        else:
            pass
    netlist.writelines('\n')

    netlist_footer = open('./netlistFooter', 'w')
    netlist_footer.writelines('// Footer End')

    netlist_footer.close()
    netlist.close()
    netlist_footer.close()

    os.chdir('..')

    # generate Ocean script for simulation
    generate_simScript()

def generate_simScript():
    script = open('./sim_opamp.ocn', 'w')
    script.writelines('simulator( \'spectre )\n')
    script.writelines('design(	 \"./netlist/netlist\")\n')
    script.writelines('resultsDir( \"./sim_results\" )\n')
    script.writelines('modelFile( \n')
    for section in model_sections:
        script.writelines('    \'(\"{}\" "{}")\n'.format(model_path, section))
    script.writelines(')\n')
    script.writelines('analysis(\'dc ?saveOppoint t  )\n') # perform dc simulation
    script.writelines('analysis(\'ac ?start \"1\"  ?stop \"0.1G\"  ?dec \"100\"  )\n') # perform ac simulation
    script.writelines('envOption(	\'analysisOrder  list(\"dc\" \"ac\") )\n')
    script.writelines('option( ?categ \'turboOpts        \'uniMode  \"APS\")\n')
    script.writelines('saveOption( \'save \"all\" )\n')
    script.writelines('temp( 27 )\n')
    script.writelines('run()\n')
    script.writelines('gain = value(db(vfreq(\'ac \"{}\")) 1)\n'.format(node_vo))
    script.writelines('pm = phaseMargin(VF(\"{}\"))\n'.format(node_vo))
    script.writelines('gbw = cross(db(vfreq(\'ac \"{}\")) 0 1 \"either\" nil nil)\n'.format(node_vo)) #script.writelines('gbw = unityGainFreq(db(vfreq(\'ac \"{}\")))\n'.format(node_vo)) 
    script.writelines('pw = abs(OP("V1" "pwr"))\n')
    # write out simulation results && exit
    script.writelines('ocnPrint(?output \"./sim.out\", ?precision 16 ?width 20 ?numberNotation \'scientific gain pm gbw pw)\n')
    script.writelines('exit\n')

def generate_netlist_lvs(intrinsic_gain_times, pd_data, debug=None):
    # for lvs, we must generate cdl format netlist here
    if debug==None:
        debug = False
    # build LUT
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    netlist = open('./netlist_lvs', 'w')
    circuit_name = pd_data.loc['circuit_name'][0]
    local_time = get_time()

    netlist.writelines('****************************************************************************\n')
    netlist.writelines('* auCdl Netlist:\n')
    netlist.writelines('* \n')
    netlist.writelines('* Library Name: OPAMP\n')
    netlist.writelines('* Top Cell Name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlist.writelines('* View Name: schematic\n')
    netlist.writelines('* Netlisted on: {}\n'.format(local_time))
    netlist.writelines('****************************************************************************\n')
    netlist.writelines('*.BIPOLAR\n*.RESI = 2000\n*.RESVAL\n*.CAPVAL\n*.DIOPERI\n*.DIOAREA\n*.EQUATION\n*.SCALE METER\n*.MEGA\n')
    netlist.writelines('\n')
    netlist.writelines('\n')

    # write gm subckts
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)
    if debug:
        print('[Debug] Invalid RC port detected: {}'.format(invalid_port))
    gm_index = 0

    netlist.writelines('****************************************************************************\n')
    netlist.writelines('* Library Name: OPAMP\n')
    netlist.writelines('* Cell Name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlist.writelines('* View Name: schematic\n')
    netlist.writelines('****************************************************************************\n')
    netlist.writelines('.SUBCKT OPAMP_{} VDD GND NET_VIP NET_VIN NET_VO NET_VB_N NET_VB_P\n'.format(circuit_name.upper()))

    # set the differential input
    negative_input = 'NET_VIN'
    positive_input = 'NET_VIP'

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    inst_index      = 0
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst].upper()
        right_port = pd_data.loc['right_port'][inst].upper()
        inst_unit = pd_data.loc['unit'][inst]
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            if (left_port == '0') or (left_port =='vss'):
                left_port = 'GND'
            if (right_port == '0') or (right_port =='vss'):
                right_port = 'GND'
            if left_port == 'vdd!':
                left_port = 'VDD'
            if right_port == 'vdd!':
                right_port = 'VDD'
            l, w, c_new = get_cap_size(inst_value, inst_unit, left_port, right_port)
            netlist.writelines('C_{} {} {} {}{} m=1 $.MODEL=M5\n'.format(capacitor_index, left_port, right_port, inst_value, inst_unit))
            capacitor_index += 1
        elif inst_type == 'resistor':
            if (left_port == '0') or (left_port =='vss'):
                left_port = 'GND'
            if (right_port == '0') or (right_port =='vss'):
                right_port = 'GND'
            if left_port == 'vdd!':
                left_port = 'VDD'
            if right_port == 'vdd!':
                right_port = 'VDD'
            l, w, seg, r_new = get_res_size(inst_value, inst_unit)
            netlist.writelines('R_{} {} {} {}{} m=1 $.MODEL=LR\n'.format(resistor_index, left_port, right_port, inst_value, inst_unit))
            resistor_index += 1
        elif inst_type == 'vccs':
            times = map_unit(inst_unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            if left_port == 'NET_VIN':
                if gm_cal > 0:
                    gmidTarget_1 = None
                    gmidTarget_2 = None
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_positive_diff_lvs(netlist, gm_index, 'VDD', negative_input, positive_input, right_port, 'GND', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                else:
                    gmidTarget_1 = None
                    gmidTarget_2 = None
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_diff(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_negative_diff_lvs(netlist, gm_index, "VDD", negative_input, positive_input, right_port, 'GND', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
            else:
                if gm_cal > 0:
                    gmidTarget_1 = None
                    gmidTarget_2 = None
                    W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_positive_mid_lvs(netlist, gm_index, 'VDD', left_port, right_port, 'GND', 'NET_VB_N', W2, L2, M2, W1, L1, M1, W0, L0, M0)
                else:
                    gmidTarget_1 = None
                    gmidTarget_2 = None
                    W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_mid(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_negative_mid_lvs(netlist, gm_index, 'VDD', left_port, right_port, 'GND', 'NET_VB_P', W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
            gm_index += 1
        else:
            pass
    netlist.writelines('.ENDS\n')
    netlist.writelines('\n')
    netlist.writelines('\n')

def generate_netlist_lvs_opt_gmid(intrinsic_gain_times, pd_data, gmid_param, debug=None):
    # for lvs, we must generate cdl format netlist here
    if debug==None:
        debug = False
    # build LUT
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    netlist = open('./netlist_lvs', 'w')
    circuit_name = pd_data.loc['circuit_name'][0]
    local_time = get_time()

    netlist.writelines('****************************************************************************\n')
    netlist.writelines('* auCdl Netlist:\n')
    netlist.writelines('* \n')
    netlist.writelines('* Library Name: OPAMP\n')
    netlist.writelines('* Top Cell Name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlist.writelines('* View Name: schematic\n')
    netlist.writelines('* Netlisted on: {}\n'.format(local_time))
    netlist.writelines('****************************************************************************\n')
    netlist.writelines('*.BIPOLAR\n*.RESI = 2000\n*.RESVAL\n*.CAPVAL\n*.DIOPERI\n*.DIOAREA\n*.EQUATION\n*.SCALE METER\n*.MEGA\n')
    netlist.writelines('\n')
    netlist.writelines('\n')

    # write gm subckts
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)
    if debug:
        print('[Debug] Invalid RC port detected: {}'.format(invalid_port))
    gm_index = 0

    netlist.writelines('****************************************************************************\n')
    netlist.writelines('* Library Name: OPAMP\n')
    netlist.writelines('* Cell Name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlist.writelines('* View Name: schematic\n')
    netlist.writelines('****************************************************************************\n')
    netlist.writelines('.SUBCKT OPAMP_{} VDD GND NET_VIP NET_VIN NET_VO NET_VB_N NET_VB_P\n'.format(circuit_name.upper()))

    # set the differential input
    negative_input = 'NET_VIN'
    positive_input = 'NET_VIP'

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    inst_index      = 0
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst].upper()
        right_port = pd_data.loc['right_port'][inst].upper()
        inst_unit = pd_data.loc['unit'][inst]
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            if (left_port == '0') or (left_port =='VSS'):
                left_port = 'GND'
            if (right_port == '0') or (right_port =='VSS'):
                right_port = 'GND'
            if left_port == 'VDD!':
                left_port = 'VDD'
            if right_port == 'VDD!':
                right_port = 'VDD'
            l, w, c_new = get_cap_size(inst_value, inst_unit, left_port, right_port)
            netlist.writelines('C_{} {} {} {}{} m=1 $.MODEL=M5\n'.format(capacitor_index, left_port, right_port, c_new, inst_unit))
            capacitor_index += 1
        elif inst_type == 'resistor':
            if (left_port == '0') or (left_port =='VSS'):
                left_port = 'GND'
            if (right_port == '0') or (right_port =='VSS'):
                right_port = 'GND'
            if left_port == 'VDD!':
                left_port = 'VDD'
            if right_port == 'VDD!':
                right_port = 'VDD'
            l, w, seg, r_new = get_res_size(inst_value, inst_unit)
            netlist.writelines('R_{} {} {} {}{} m=1 $.MODEL=LR\n'.format(resistor_index, left_port, right_port, r_new, inst_unit))
            resistor_index += 1
        elif inst_type == 'vccs':
            times = map_unit(inst_unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            if left_port == 'NET_VIN':
                if gm_cal > 0:
                    gmidTarget_1 = gmid_param['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = gmid_param['gmid_2_'+str(gm_index)][0]
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_positive_diff_lvs(netlist, gm_index, 'VDD', negative_input, positive_input, right_port, 'GND', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                else:
                    gmidTarget_1 = gmid_param['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = gmid_param['gmid_2_'+str(gm_index)][0]
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_diff(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_negative_diff_lvs(netlist, gm_index, "VDD", negative_input, positive_input, right_port, 'GND', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
            else:
                if gm_cal > 0:
                    gmidTarget_1 = gmid_param['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = gmid_param['gmid_2_'+str(gm_index)][0]
                    W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_positive_mid_lvs(netlist, gm_index, 'VDD', left_port, right_port, 'GND', 'NET_VB_N', W2, L2, M2, W1, L1, M1, W0, L0, M0)
                else:
                    gmidTarget_1 = gmid_param['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = gmid_param['gmid_2_'+str(gm_index)][0]
                    W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_mid(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gm_name = get_gm_negative_mid_lvs(netlist, gm_index, 'VDD', left_port, right_port, 'GND', 'NET_VB_P', W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
            gm_index += 1
        else:
            pass
    netlist.writelines('.ENDS\n')
    netlist.writelines('\n')
    netlist.writelines('\n')

def generate_netlistOut(intrinsic_gain_times, pd_data):
    all_param_dict = {}
    gmid_target_dict = {}
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)

    os.system('mkdir netlist_Out')
    os.chdir('./netlist_Out')

    circuit_name = pd_data.loc['circuit_name'][0]

    manualcons_file = open('manualcons','w') # constraint file
    M_index = 1 # match groups
    S_index = 1 # symmetry groups
    P_index = 1 # proximity groups

    netlistOut = open('netlistout','w') # netlistout file
    netlistOut.writelines('lib_name: OPAMP\n')
    netlistOut.writelines('cell_name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlistOut.writelines('pins: ["VDD", "GND", "NET_VIP", "NET_VIN", "NET_VO", "NET_VB_N", "NET_VB_P"]\n')
    netlistOut.writelines('instances:\n')

    netlistOut_file = open('netlistout.simp','w') # netlistout.simp file
    netlistOut_file.writelines('cell_name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlistOut_file.writelines('instances:\n')

    # set the differential input
    negative_input = 'NET_VIN'
    positive_input = 'NET_VIP'

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    gm_index        = 0
    r_list          = []
    c_list          = []
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst].upper()
        right_port = pd_data.loc['right_port'][inst].upper()
        inst_unit = pd_data.loc['unit'][inst]
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            c_list.append('C_'+str(capacitor_index))
            gen_c_out(netlistOut, netlistOut_file, 'C_'+str(capacitor_index), left_port, right_port, inst_value, inst_unit)
            capacitor_index += 1
        elif inst_type == 'resistor':
            r_list.append('R_'+str(resistor_index))
            gen_r_out(netlistOut, netlistOut_file, 'R_'+str(resistor_index), left_port, right_port, inst_value, inst_unit)
            resistor_index += 1
        elif inst_type == 'vccs':
            times = map_unit(inst_unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            if left_port == 'NET_VIN':
                if gm_cal > 0:
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    gen_positive_diff_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', negative_input, positive_input, right_port, 'vss', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gmid_target_dict.update({'gmid_1_'+str(gm_index):15})
                    gmid_target_dict.update({'gmid_2_'+str(gm_index):8})
                    M_index += 3
                    S_index += 1
                else:
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_diff(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    gen_negative_diff_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, "vdd!", negative_input, positive_input, right_port, 'vss', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gmid_target_dict.update({'gmid_1_'+str(gm_index):15})
                    gmid_target_dict.update({'gmid_2_'+str(gm_index):8})
                    M_index += 3
                    S_index += 1
            else:
                if gm_cal > 0:
                    W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    gen_positive_mid_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', left_port, right_port, 'vss', 'NET_VB_N', W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gmid_target_dict.update({'gmid_1_'+str(gm_index):15})
                    gmid_target_dict.update({'gmid_2_'+str(gm_index):8})
                    S_index += 1
                    M_index += 1
                else:
                    W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_mid(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    gen_negative_mid_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', left_port, right_port, 'vss', 'NET_VB_P', W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    gmid_target_dict.update({'gmid_1_'+str(gm_index):15})
                    gmid_target_dict.update({'gmid_2_'+str(gm_index):8})
                    M_index += 3
                    S_index += 1
            gm_index += 1
        else:
            pass
    
    netlistOut_file.writelines('lib_name: OPAMP\n')
    netlistOut_file.writelines('pins:\n- VDD\n- GND\n- NET_VIP\n- NET_VIN\n- NET_VO\n- NET_VB_N\n- NET_VB_P\n')
    
    pins_list = ['PIN0', 'PIN1', 'PIN2', 'PIN3', 'PIN4', 'PIN5', 'PIN6']
    for pin in pins_list:
        netlistOut.writelines('  {}:\n'.format(pin))
        netlistOut.writelines('    lib_name: basic\n')
        netlistOut.writelines('    cell_name: iopin\n')
        netlistOut.writelines('    instpins: {}\n')
        netlistOut.writelines('    master_prop:\n')
        netlistOut.writelines('    inst_prop:\n')
    
    S = 1
    manualcons_file.writelines('M{} '.format(M_index))
    while S< S_index:
        manualcons_file.writelines('S{} '.format(S))
        S += 1
    manualcons_file.writelines('\n')
    manualcons_file.writelines('M{} '.format(M_index+1))
    for r in r_list:
        manualcons_file.writelines('OPAMP_{}_{} '.format(circuit_name.upper(), r))
    for c in c_list:
        manualcons_file.writelines('OPAMP_{}_{} '.format(circuit_name.upper(), c))
    manualcons_file.writelines('\n')
    manualcons_file.writelines('S{} M{} M{} \n'.format(S_index, M_index, M_index+1))
    
    netlistOut.close()
    netlistOut_file.close()
    manualcons_file.close()

    generate_netlist_lvs(intrinsic_gain_times, pd_data, debug=None) # netlist file for lvs

    return all_param_dict, gmid_target_dict

def generate_netlistOut_opt(pd_data, param):
    # TODO
    all_param_dict = {}
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)

    os.system('mkdir netlist_Out_opt')
    os.chdir('./netlist_Out_opt')
    
    generate_netlist_lvs(pd_data, debug=None) # netlist file for lvs

    circuit_name = pd_data.loc['circuit_name'][0]

    manualcons_file = open('manualcons','w') # constraint file
    M_index = 1 # match groups
    S_index = 1 # symmetry groups
    P_index = 1 # proximity groups

    netlistOut = open('netlistout','w') # netlistout file
    netlistOut.writelines('lib_name: opamp\n')
    netlistOut.writelines('cell_name: opamp_{}\n'.format(circuit_name))
    netlistOut.writelines('pins: ["VDD", "GND", "net_vip", "net_vin", "net_vo", "net_vb_n" "net_vb_p"]\n')
    netlistOut.writelines('instances:\n')

    netlistOut_file = open('netlistout.simp','w') # netlistout.simp file
    netlistOut_file.writelines('cell_name: opamp_{}\n'.format(circuit_name))
    netlistOut_file.writelines('instances:\n')

    # set the differential input
    negative_input = node_vin
    positive_input = 'net_vip'

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    gm_index        = 0
    r_list          = []
    c_list          = []
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst]
        right_port = pd_data.loc['right_port'][inst]
        inst_unit = pd_data.loc['unit'][inst]
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            c_list.append('C_'+str(capacitor_index))
            gen_c_out(netlistOut, netlistOut_file, 'C_'+str(capacitor_index), left_port, right_port, inst_value, inst_unit)
            capacitor_index += 1
        elif inst_type == 'resistor':
            r_list.append('R_'+str(resistor_index))
            gen_r_out(netlistOut, netlistOut_file, 'R_'+str(resistor_index), left_port, right_port, inst_value, inst_unit)
            resistor_index += 1
        elif inst_type == 'vccs':
            times = map_unit(inst_unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            if left_port == node_vin:
                if gm_cal > 0:
                    _, L5, M5, _, L4, M4, _, L3, M3, _, L2, M2, _, L1, M1, _, L0, M0 = cal_positive_diff(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W5 = param['M5_'+str(gm_index)+'_w'][0]
                    W4 = param['M4_'+str(gm_index)+'_w'][0]
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W3 = W2
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    W1 = W0
                    gen_positive_diff_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', negative_input, positive_input, right_port, 'vss', "net_vb_n", W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    M_index += 3
                    S_index += 1
                else:
                    _, L5, M5, _, L4, M4, _, L3, M3, _, L2, M2, _, L1, M1, _, L0, M0 = cal_negative_diff(-1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W5 = param['M5_'+str(gm_index)+'_w'][0]
                    W4 = param['M4_'+str(gm_index)+'_w'][0]
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W3 = W2
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    W1 = W0
                    gen_negative_diff_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, "vdd!", negative_input, positive_input, right_port, 'vss', "net_vb_n", W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    M_index += 3
                    S_index += 1
            else:
                if gm_cal > 0:
                    _, L2, M2, _, L1, M1, _, L0, M0 = cal_positive_mid(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W1 = param['M1_'+str(gm_index)+'_w'][0]
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    gen_positive_mid_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', left_port, right_port, 'vss', "net_vb_n", W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    S_index += 1
                    M_index += 1
                else:
                    _, L4, M4, _, L3, M3, _, L2, M2, _, L1, M1, _, L0, M0 = cal_negative_mid(-1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
                    W4 = param['M4_'+str(gm_index)+'_w'][0]
                    W3 = param['M3_'+str(gm_index)+'_w'][0]
                    W2 = param['M2_'+str(gm_index)+'_w'][0]
                    W0 = param['M0_'+str(gm_index)+'_w'][0]
                    W1 = W0
                    gen_negative_mid_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', left_port, right_port, 'vss', "net_vb_p", W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    M_index += 3
                    S_index += 1
            gm_index += 1
        else:
            pass
    
    netlistOut_file.writelines('lib_name: OPAMP\n')
    netlistOut_file.writelines('pins:\n- VDD\n- GND\n- net_vip\n- net_vin\n- net_vo\n- net_vb_n\n- net_vb_p\n')
    
    pins_list = ['PIN0', 'PIN1', 'PIN2', 'PIN3', 'PIN4', 'PIN5', 'PIN6']
    for pin in pins_list:
        netlistOut.writelines('  {}:\n'.format(pin))
        netlistOut.writelines('    lib_name: basic\n')
        netlistOut.writelines('    cell_name: iopin\n')
        netlistOut.writelines('    instpins: {}\n')
        netlistOut.writelines('    master_prop:\n')
        netlistOut.writelines('    inst_prop:\n')
    
    S = 1
    manualcons_file.writelines('M{} '.format(M_index))
    while S< S_index:
        manualcons_file.writelines('S{} '.format(S))
        S += 1
    manualcons_file.writelines('\n')
    manualcons_file.writelines('M{} '.format(M_index+1))
    for r in r_list:
        manualcons_file.writelines('OPAMP_{}_{} '.format(circuit_name.upper(), r))
    for c in c_list:
        manualcons_file.writelines('OPAMP_{}_{} '.format(circuit_name.upper(), c))
    manualcons_file.writelines('\n')
    manualcons_file.writelines('S{} M{} M{} \n'.format(S_index, M_index, M_index+1))
    
    netlistOut.close()
    netlistOut_file.close()
    manualcons_file.close()

    return all_param_dict

def generate_netlistOut_opt_gmid(intrinsic_gain_times, pd_data, param_gmid):
    all_param_dict = {}
    gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    inst_list = pd_data.columns.values.tolist()
    instance_list, invalid_port = remove_invalid_device(inst_list, pd_data)

    os.system('mkdir netlist_Out_opt_gmid')
    os.chdir('./netlist_Out_opt_gmid')

    circuit_name = pd_data.loc['circuit_name'][0]

    manualcons_file = open('manualcons','w') # constraint file
    M_index = 1 # match groups
    S_index = 1 # symmetry groups
    P_index = 1 # proximity groups

    netlistOut = open('netlistout','w') # netlistout file
    netlistOut.writelines('lib_name: OPAMP\n')
    netlistOut.writelines('cell_name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlistOut.writelines('pins: ["VDD", "GND", "NET_VIP", "NET_VIN", "NET_VO", "NET_VB_N", "NET_VB_P"]\n')
    netlistOut.writelines('instances:\n')

    netlistOut_file = open('netlistout.simp','w') # netlistout.simp file
    netlistOut_file.writelines('cell_name: OPAMP_{}\n'.format(circuit_name.upper()))
    netlistOut_file.writelines('instances:\n')

    # set the differential input
    negative_input = 'NET_VIN'
    positive_input = 'NET_VIP'

    # write all instances
    capacitor_index = 2
    resistor_index  = 0
    gm_index        = 0
    r_list          = []
    c_list          = []
    for inst in instance_list:
        inst_type = pd_data.loc['type'][inst]
        left_port = pd_data.loc['left_port'][inst].upper()
        right_port = pd_data.loc['right_port'][inst].upper()
        inst_unit = pd_data.loc['unit'][inst]
        inst_value = pd_data.loc['value'][inst]
        parasitic = pd_data.loc['parasitic'][inst]
        if inst_type == 'capacitor':
            c_list.append('C_'+str(capacitor_index))
            gen_c_out(netlistOut, netlistOut_file, 'C_'+str(capacitor_index), left_port, right_port, inst_value, inst_unit)
            capacitor_index += 1
        elif inst_type == 'resistor':
            r_list.append('R_'+str(resistor_index))
            gen_r_out(netlistOut, netlistOut_file, 'R_'+str(resistor_index), left_port, right_port, inst_value, inst_unit)
            resistor_index += 1
        elif inst_type == 'vccs':
            times = map_unit(inst_unit)
            gm_cal = pd_data.loc['value'][inst] * times
            gds_cal = get_gds(inst, pd_data)
            if left_port == 'NET_VIN':
                if gm_cal > 0:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gen_positive_diff_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', negative_input, positive_input, right_port, 'vss', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    M_index += 3
                    S_index += 1
                else:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_diff(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gen_negative_diff_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, "vdd!", negative_input, positive_input, right_port, 'vss', 'NET_VB_N', W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    M_index += 3
                    S_index += 1
            else:
                if gm_cal > 0:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gen_positive_mid_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', left_port, right_port, 'vss', 'NET_VB_N', W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    S_index += 1
                    M_index += 1
                else:
                    gmidTarget_1 = param_gmid['gmid_1_'+str(gm_index)][0]
                    gmidTarget_2 = param_gmid['gmid_2_'+str(gm_index)][0]
                    W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0 = cal_negative_mid(intrinsic_gain_times, -1*gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, gmid_target_1=gmidTarget_1, gmid_target_2=gmidTarget_2)
                    gen_negative_mid_out(circuit_name, all_param_dict, M_index, S_index, manualcons_file, netlistOut, netlistOut_file, gm_index, 'vdd!', left_port, right_port, 'vss', 'NET_VB_P', W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0)
                    M_index += 3
                    S_index += 1
            gm_index += 1
        else:
            pass
    
    netlistOut_file.writelines('lib_name: OPAMP\n')
    netlistOut_file.writelines('pins:\n- VDD\n- GND\n- NET_VIP\n- NET_VIN\n- NET_VO\n- NET_VB_N\n- NET_VB_P\n')
    
    pins_list = ['PIN0', 'PIN1', 'PIN2', 'PIN3', 'PIN4', 'PIN5', 'PIN6']
    for pin in pins_list:
        netlistOut.writelines('  {}:\n'.format(pin))
        netlistOut.writelines('    lib_name: basic\n')
        netlistOut.writelines('    cell_name: iopin\n')
        netlistOut.writelines('    instpins: {}\n')
        netlistOut.writelines('    master_prop:\n')
        netlistOut.writelines('    inst_prop:\n')
    
    S = 1
    manualcons_file.writelines('M{} '.format(M_index))
    while S< S_index:
        manualcons_file.writelines('S{} '.format(S))
        S += 1
    manualcons_file.writelines('\n')
    manualcons_file.writelines('M{} '.format(M_index+1))
    for r in r_list:
        manualcons_file.writelines('OPAMP_{}_{} '.format(circuit_name.upper(), r))
    for c in c_list:
        manualcons_file.writelines('OPAMP_{}_{} '.format(circuit_name.upper(), c))
    manualcons_file.writelines('\n')
    manualcons_file.writelines('S{} M{} M{} \n'.format(S_index, M_index, M_index+1))
    
    netlistOut.close()
    netlistOut_file.close()
    manualcons_file.close()

    generate_netlist_lvs_opt_gmid(intrinsic_gain_times, pd_data, gmid_param=param_gmid, debug=None) # netlist file for lvs

    return all_param_dict

if __name__ == '__main__':
    """check simulation script generation"""
    # generate_simScript()

    """check the size cal"""
    # gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)
    
    # W5, L5, W4, L4, W3, L3, W2, L2, W1, L1, W0, L0 = cal_positive_diff(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
    # print(W5, L5, W4, L4, W3, L3, W2, L2, W1, L1, W0, L0)
    # W2, L2, W1, L1, W0, L0 = cal_positive_mid(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
    # print(W2, L2, W1, L1, W0, L0)
    # W5, L5, W4, L4, W3, L3, W2, L2, W1, L1, W0, L0 = cal_negative_diff(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
    # print(W5, L5, W4, L4, W3, L3, W2, L2, W1, L1, W0, L0)
    # W4, L4, W3, L3, W2, L2, W1, L1, W0, L0 = cal_negative_mid(gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p)
    # print(W4, L4, W3, L3, W2, L2, W1, L1, W0, L0)

    """check the LUT building"""
    # gm_n, gds_n, id_n, gm_p, gds_p, id_p = build_lut(nmos_lut_path, pmos_lut_path)

    """check the raw data"""
    gm_nmos = pd.read_csv(nmos_lut_path+'gm.csv')
    gm_nmos = replace_col(gm_nmos, L_list, ['_vgs', '_gm'])
    ro_nmos = pd.read_csv(nmos_lut_path+'ro.csv') 
    ro_nmos = replace_col(ro_nmos, L_list, ['_vgs', '_ro'])
    id_nmos = pd.read_csv(nmos_lut_path+'id.csv')
    id_nmos = replace_col(id_nmos, L_list, ['_vgs', '_id'])

    gm_pmos = pd.read_csv(nmos_lut_path+'gm.csv')
    gm_pmos = replace_col(gm_pmos, L_list, ['_vgs', '_gm'])
    ro_pmos = pd.read_csv(nmos_lut_path+'ro.csv')
    ro_pmos = replace_col(ro_pmos, L_list, ['_vgs', '_ro'])
    id_pmos = pd.read_csv(nmos_lut_path+'id.csv')
    id_pmos = replace_col(id_pmos, L_list, ['_vgs', '_id'])

    L_test = [1.8e-07, 2.07437185929648e-07, 4.08643216080402e-07, 6.0070351758794e-07, 8.01909547738694e-07, 1.00311557788945e-06, 1.2043216080402e-06, 1.40552763819095e-06, 1.6067336683417e-06, 1.80793969849245e-06, 2e-06]

    fig = plt.figure()
    for l in L_test:
        gm  = gm_nmos[str(l)+'_gm'].values.tolist()
        ro  = ro_nmos[str(l)+'_ro'].values.tolist()
        id  = id_nmos[str(l)+'_id'].values.tolist()
        plt.xticks([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        plt.plot(id, list(np.array(gm)/np.array(id)))
        plt.xscale('log')
        plt.grid(linestyle='-.')
    plt.savefig('./id.png')

    fig = plt.figure()
    for l in L_test:
        gm  = gm_nmos[str(l)+'_gm'].values.tolist()
        ro  = ro_nmos[str(l)+'_ro'].values.tolist()
        id  = id_nmos[str(l)+'_id'].values.tolist()
        plt.plot(list(np.array(gm)/np.array(id)), list(np.array(gm)/np.array(ro)))
        plt.grid(linestyle='-.')
    plt.savefig('./gm_gds.png')

    # fig = plt.figure()
    # for l in L_list:
    #     vgs = gm_nmos[str(l)+'_vgs'].values.tolist()
    #     gm  = gm_nmos[str(l)+'_gm'].values.tolist()
    #     plt.plot(vgs, gm)
    # plt.xlabel('VGS(V)')
    # plt.ylabel('gm(S)')
    # plt.savefig('./gm.png')

    # fig = plt.figure()
    # for l in L_list:
    #     vgs = ro_nmos[str(l)+'_vgs'].values.tolist()
    #     ro  = ro_nmos[str(l)+'_ro'].values.tolist()
    #     plt.plot(vgs, ro)
    # plt.xlabel('VGS(V)')
    # plt.ylabel('ro(S)')
    # plt.savefig('./ro.png')

    # fig = plt.figure()
    # for l in L_list:
    #     vgs = id_nmos[str(l)+'_vgs'].values.tolist()
    #     id  = id_nmos[str(l)+'_id'].values.tolist()
    #     plt.plot(vgs, id)
    # plt.xlabel('VGS(V)')
    # plt.ylabel('id(A)')
    # plt.savefig('./id.png')
