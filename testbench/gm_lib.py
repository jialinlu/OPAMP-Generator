# To get different gms in transistor level, as well as the netlist out file (connections and constraints)
# Detail gms' topology plz refer to my document
# Based on the netlist generated from virtuoso
# Netlists are related to target PDK, here is generated from tsmc18rf

import math
import numpy as np
from scipy import interpolate

# L is 200 points sampled from [0.18u, 2u], because the PDK used is tsmc18rf
L_list = [1.8e-07, 1.89145728643216e-07, 1.98291457286432e-07, 2.07437185929648e-07, 2.16582914572864e-07, 2.2572864321608e-07, 2.34874371859296e-07, 2.44020100502512e-07, 2.53165829145729e-07, 2.62311557788945e-07, 2.71457286432161e-07, 2.80603015075377e-07, 2.89748743718593e-07, 2.98894472361809e-07, 3.08040201005025e-07, 3.17185929648241e-07, 3.26331658291457e-07, 3.35477386934673e-07, 3.44623115577889e-07, 3.53768844221106e-07, 3.62914572864322e-07, 3.72060301507538e-07, 3.81206030150754e-07, 3.9035175879397e-07, 3.99497487437186e-07, 4.08643216080402e-07, 4.17788944723618e-07, 4.26934673366834e-07, 4.3608040201005e-07, 4.45226130653266e-07, 4.54371859296482e-07, 4.63517587939699e-07, 4.72663316582915e-07, 4.81809045226131e-07, 4.90954773869347e-07, 5.00100502512563e-07, 5.09246231155779e-07, 5.18391959798995e-07, 5.27537688442211e-07, 5.36683417085427e-07, 5.45829145728643e-07, 5.54974874371859e-07, 5.64120603015075e-07, 5.73266331658291e-07, 5.82412060301508e-07, 5.91557788944724e-07, 6.0070351758794e-07, 6.09849246231156e-07, 6.18994974874372e-07, 6.28140703517588e-07, 6.37286432160804e-07, 6.4643216080402e-07, 6.55577889447236e-07, 6.64723618090452e-07, 6.73869346733668e-07, 6.83015075376884e-07, 6.92160804020101e-07, 7.01306532663317e-07, 7.10452261306533e-07, 7.19597989949749e-07, 7.28743718592965e-07, 7.37889447236181e-07, 7.47035175879397e-07, 7.56180904522613e-07, 7.65326633165829e-07, 7.74472361809045e-07, 7.83618090452261e-07, 7.92763819095477e-07, 8.01909547738694e-07, 8.1105527638191e-07, 8.20201005025126e-07, 8.29346733668342e-07, 8.38492462311558e-07, 8.47638190954774e-07, 8.5678391959799e-07, 8.65929648241206e-07, 8.75075376884422e-07, 8.84221105527638e-07, 8.93366834170854e-07, 9.0251256281407e-07, 9.11658291457287e-07, 9.20804020100503e-07, 9.29949748743719e-07, 9.39095477386935e-07, 9.48241206030151e-07, 9.57386934673367e-07, 9.66532663316583e-07, 9.75678391959799e-07, 9.84824120603015e-07, 9.93969849246231e-07, 1.00311557788945e-06, 1.01226130653266e-06, 1.02140703517588e-06, 1.0305527638191e-06, 1.03969849246231e-06, 1.04884422110553e-06, 1.05798994974874e-06, 1.06713567839196e-06, 1.07628140703517e-06, 1.08542713567839e-06, 1.09457286432161e-06, 1.10371859296482e-06, 1.11286432160804e-06, 1.12201005025125e-06, 1.13115577889447e-06, 1.14030150753769e-06, 1.1494472361809e-06, 1.15859296482412e-06, 1.16773869346733e-06, 1.17688442211055e-06, 1.18603015075377e-06, 1.19517587939698e-06, 1.2043216080402e-06, 1.21346733668341e-06, 1.22261306532663e-06, 1.23175879396985e-06, 1.24090452261306e-06, 1.25005025125628e-06, 1.25919597989949e-06, 1.26834170854271e-06, 1.27748743718593e-06, 1.28663316582914e-06, 1.29577889447236e-06, 1.30492462311557e-06, 1.31407035175879e-06, 1.32321608040201e-06, 1.33236180904522e-06, 1.34150753768844e-06, 1.35065326633165e-06, 1.35979899497487e-06, 1.36894472361809e-06, 1.3780904522613e-06, 1.38723618090452e-06, 1.39638190954773e-06, 1.40552763819095e-06, 1.41467336683417e-06, 1.42381909547738e-06, 1.4329648241206e-06, 1.44211055276381e-06, 1.45125628140703e-06, 1.46040201005025e-06, 1.46954773869346e-06, 1.47869346733668e-06, 1.48783919597989e-06, 1.49698492462311e-06, 1.50613065326632e-06, 1.51527638190954e-06, 1.52442211055276e-06, 1.53356783919597e-06, 1.54271356783919e-06, 1.55185929648241e-06, 1.56100502512562e-06, 1.57015075376884e-06, 1.57929648241205e-06, 1.58844221105527e-06, 1.59758793969848e-06, 1.6067336683417e-06, 1.61587939698492e-06, 1.62502512562813e-06, 1.63417085427135e-06, 1.64331658291456e-06, 1.65246231155778e-06, 1.661608040201e-06, 1.67075376884421e-06, 1.67989949748743e-06, 1.68904522613065e-06, 1.69819095477386e-06, 1.70733668341708e-06, 1.71648241206029e-06, 1.72562814070351e-06, 1.73477386934672e-06, 1.74391959798994e-06,1.75306532663316e-06, 1.76221105527637e-06, 1.77135678391959e-06, 1.78050251256281e-06, 1.78964824120602e-06, 1.79879396984924e-06, 1.80793969849245e-06, 1.81708542713567e-06, 1.82623115577888e-06, 1.8353768844221e-06, 1.84452261306532e-06, 1.85366834170853e-06, 1.86281407035175e-06, 1.87195979899496e-06, 1.88110552763818e-06, 1.8902512562814e-06, 1.89939698492461e-06, 1.90854271356783e-06, 1.91768844221104e-06, 1.92683417085426e-06, 1.93597989949748e-06, 1.94512562814069e-06, 1.95427135678391e-06, 1.96341708542712e-06, 1.97256281407034e-06, 1.98170854271356e-06, 1.99085427135677e-06, 2e-06]
#L_list = [8.20201005025126e-07, 8.29346733668342e-07, 8.38492462311558e-07, 8.47638190954774e-07, 8.5678391959799e-07, 8.65929648241206e-07, 8.75075376884422e-07, 8.84221105527638e-07, 8.93366834170854e-07, 9.0251256281407e-07, 9.11658291457287e-07, 9.20804020100503e-07, 9.29949748743719e-07, 9.39095477386935e-07, 9.48241206030151e-07, 9.57386934673367e-07, 9.66532663316583e-07, 9.75678391959799e-07, 9.84824120603015e-07, 9.93969849246231e-07, 1.00311557788945e-06, 1.01226130653266e-06, 1.02140703517588e-06, 1.0305527638191e-06, 1.03969849246231e-06, 1.04884422110553e-06, 1.05798994974874e-06, 1.06713567839196e-06, 1.07628140703517e-06, 1.08542713567839e-06, 1.09457286432161e-06, 1.10371859296482e-06, 1.11286432160804e-06, 1.12201005025125e-06, 1.13115577889447e-06, 1.14030150753769e-06, 1.1494472361809e-06, 1.15859296482412e-06, 1.16773869346733e-06, 1.17688442211055e-06, 1.18603015075377e-06, 1.19517587939698e-06, 1.2043216080402e-06, 1.21346733668341e-06, 1.22261306532663e-06, 1.23175879396985e-06, 1.24090452261306e-06, 1.25005025125628e-06, 1.25919597989949e-06, 1.26834170854271e-06, 1.27748743718593e-06, 1.28663316582914e-06, 1.29577889447236e-06, 1.30492462311557e-06, 1.31407035175879e-06, 1.32321608040201e-06, 1.33236180904522e-06, 1.34150753768844e-06, 1.35065326633165e-06, 1.35979899497487e-06, 1.36894472361809e-06, 1.3780904522613e-06, 1.38723618090452e-06, 1.39638190954773e-06, 1.40552763819095e-06, 1.41467336683417e-06, 1.42381909547738e-06, 1.4329648241206e-06, 1.44211055276381e-06, 1.45125628140703e-06, 1.46040201005025e-06, 1.46954773869346e-06, 1.47869346733668e-06, 1.48783919597989e-06, 1.49698492462311e-06, 1.50613065326632e-06, 1.51527638190954e-06, 1.52442211055276e-06, 1.53356783919597e-06, 1.54271356783919e-06, 1.55185929648241e-06, 1.56100502512562e-06, 1.57015075376884e-06, 1.57929648241205e-06, 1.58844221105527e-06, 1.59758793969848e-06, 1.6067336683417e-06, 1.61587939698492e-06, 1.62502512562813e-06, 1.63417085427135e-06, 1.64331658291456e-06, 1.65246231155778e-06, 1.661608040201e-06, 1.67075376884421e-06, 1.67989949748743e-06, 1.68904522613065e-06, 1.69819095477386e-06, 1.70733668341708e-06, 1.71648241206029e-06, 1.72562814070351e-06, 1.73477386934672e-06, 1.74391959798994e-06,1.75306532663316e-06, 1.76221105527637e-06, 1.77135678391959e-06, 1.78050251256281e-06, 1.78964824120602e-06, 1.79879396984924e-06, 1.80793969849245e-06, 1.81708542713567e-06, 1.82623115577888e-06, 1.8353768844221e-06, 1.84452261306532e-06, 1.85366834170853e-06, 1.86281407035175e-06, 1.87195979899496e-06, 1.88110552763818e-06, 1.8902512562814e-06, 1.89939698492461e-06, 1.90854271356783e-06, 1.91768844221104e-06, 1.92683417085426e-06, 1.93597989949748e-06, 1.94512562814069e-06, 1.95427135678391e-06, 1.96341708542712e-06, 1.97256281407034e-06, 1.98170854271356e-06, 1.99085427135677e-06, 2e-06]
L_array = np.array(L_list)

# tsmcrf018
lmin = 1.8e-7      # m
lmax = 20e-6       # m
wmin = 1.5e-06     # m
wmax = 900e-06     # m
cap_max_w_l = 30   # um
cap_min_w_l = 4    # um
cap_max = 951.6    # fF, one single cap
cap_min = 20.28    # fF, one single cap
res_min_w = 1      # um
res_min_l = 2      # um
res_max_w = 50     # um
res_max_l = 1000   # um
res_max = 1.6177   # Mohm, one single res
res_min = 44.7466  # ohm,  one single res

# For the size of resistors and capacitances, I fix the W of instances 
# and interpolate to get L.
# tsmcrf18
cap_l_5  = [4, 10, 15, 20, 25, 30] # w=5u
cap_v_5  = [0.0249, 0.05853, 0.08658, 0.1146, 0.1427, 0.1707] # w=5u
cap_l_18 = [5, 10, 15, 20, 25, 30] # w=18u
cap_v_18 = [0.1034, 0.1981, 0.2928, 0.3874, 0.4821, 0.5768] # w=18u
res_l_1 = [2, 10, 20, 30, 40, 50] # w=1u
res_v_1 = [0.0024121, 0.0117, 0.0233, 0.0349, 0.0465, 0.05814] # w=1u
res_l_x = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500] # w=1u, l = 50 * seg
res_v_x = [0.1163, 0.2326, 0.3489, 0.4652, 0.5814, 0.6977, 0.814, 0.93, 1.047, 1.163, 1.279, 1.3955, 1.512, 1.628, 1.744] # w=1u
res_l_x_plus = [1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000] # w=1u, l = 200 * seg
res_v_x_plus = [1.394, 1.858, 2.323, 2.788, 3.252, 3.717, 4.182, 4.646]

def get_cap_size(cap_value, cap_name, top_port, bot_port): # um
    '''estimate cap size by interpolation'''
    # if cap_unit == 'p':
    #     pass
    # else:
    #     print('[Info] convert cap value from unit {} to p'.format(cap_unit))
    #     if cap_unit == 'n':
    #         cap_value = cap_value * 1000
    #     elif cap_unit == 'f':
    #         cap_value = cap_value / 1000
    #     else:
    #         print('[Error] unsupport cap unit : {}'.format(cap_unit))
    # if cap_value<0.15:
    #     w=5
    #     f = interpolate.interp1d(cap_v_5, cap_l_5)
    #     cal_l = f(cap_value)
    #     f_x = interpolate.interp1d(cap_l_5, cap_v_5)
    #     c_tmp = f_x(cal_l)
    # else:
    #     w = 18
    #     f = interpolate.interp1d(cap_v_18, cap_l_18)
    #     cal_l = f(cap_value)
    #     f_x = interpolate.interp1d(cap_l_18, cap_v_18)
    #     c_tmp = f_x(cal_l)

    '''cap sizes lookup for known capacitance'''
    if  abs(cap_value - 0.85) < 0.01:
        cal_l = 28.3
        w = 28.3
        c_tmp = cap_value
    elif abs(cap_value - 0.669) < 0.01:
        cal_l = 25.1
        w = 25.1
        c_tmp = cap_value
    elif abs(cap_value - 0.952) < 0.01:
        cal_l = 30
        w = 30
        c_tmp = cap_value
    elif abs(cap_value - 0.5) < 0.01:
        cal_l = 21.6
        w = 21.6
        c_tmp = cap_value
    else:
        #print('[Error] unknown caps {}: {}, ports are {} {}'.format(cap_name, cap_value, top_port, bot_port))
        # just for netlistout file generated, wrong w & l
        cal_l = 30
        w = 30
        c_tmp = cap_value

    return cal_l, w, c_tmp

def get_res_size(res_value, res_name): # um
    '''estimate res size by interpolation'''
    w =1
    # if res_unit == 'M':
    #     if res_value < 0.055:
    #         f = interpolate.interp1d(res_v_1, res_l_1)
    #         cal_l = f(res_value)
    #         seg = 1
    #         f_x = interpolate.interp1d(res_l_1, res_v_1)
    #         r_tmp = res_value
    #     elif 0.055 <= res_value <= 1.7:
    #         f_x = interpolate.interp1d(res_v_x, res_l_x)
    #         l_total_tmp = f_x(res_value)
    #         seg = math.ceil(l_total_tmp/50)
    #         if (seg%2 != 0):
    #             seg += 1
    #         r_tmp = res_value/seg
    #         f = interpolate.interp1d(res_v_1, res_l_1)
    #         f_x_1 = interpolate.interp1d(res_l_1, res_v_1)
    #         cal_l = f(r_tmp)
    #         r_tmp = res_value
    #     elif 1.7<=res_value<=4.6 :
    #         f_x = interpolate.interp1d(res_v_x_plus, res_l_x_plus)
    #         l_total_tmp = f_x(res_value)
    #         seg = math.ceil(l_total_tmp/200)
    #         if (seg%2 != 0):
    #             seg += 1
    #         r_tmp = res_value/seg
    #         if r_tmp < 0.058:
    #             f = interpolate.interp1d(res_v_1, res_l_1)
    #             f_x_1 = interpolate.interp1d(res_l_1, res_v_1)
    #             cal_l = f(r_tmp)
    #             r_tmp = res_value
    #         else:
    #             f = interpolate.interp1d(res_v_x, res_l_x)
    #             f_x_1 = interpolate.interp1d(res_l_x, res_v_x)
    #             cal_l = f(r_tmp)
    #             r_tmp = res_value
    #     else :
    #         cal_l = 150
    #         seg = 60
    #         r_tmp = res_value
    # else:
    #     print('[Error] unsupport res unit: {}!'.format(res_unit))
    
    '''cap sizes lookup for known capacitance'''
    if abs(res_value-0.11) < 0.01:
        cal_l = 10
        seg = 10
    elif abs(res_value-0.794) < 0.01:
        cal_l = 26.2
        seg = 26
    elif abs(res_value-1.715) < 0.01:
        cal_l = 38.8
        seg = 38
    elif abs(res_value-3) < 0.01:
        cal_l = 52
        seg = 50
    elif abs(res_value-10) < 0.01:
        cal_l = 93.5
        seg = 92
    else:
        #print('[Error] unknown res {}: {}'.format(res_name, res_value))
        # just for netlistout file generated, wrong seg & l
        cal_l = 93.5
        seg = 92
    r_tmp = res_value
    return cal_l, w, seg, r_tmp

def gen_mos_out(tunable, param_dict, netlistOut, netlist_file, mos_name, n_or_p, d_port, g_port, s_port, b_port, l, w, m):
    if (d_port == '0') or (d_port =='vss'):
        d_port = 'GND'
    if (g_port == '0') or (g_port =='vss'):
        g_port = 'GND'
    if (s_port == '0') or (s_port =='vss'):
        s_port = 'GND'
    if (b_port == '0') or (b_port =='vss'):
        b_port = 'GND'

    if d_port == 'vdd!':
        d_port = 'VDD'
    if g_port == 'vdd!':
        g_port = 'VDD'
    if s_port == 'vdd!':
        s_port = 'VDD'
    if b_port == 'vdd!':
        b_port = 'VDD'

    # netlistout
    netlistOut.writelines('  {}:\n'.format(mos_name))
    netlistOut.writelines('    lib_name: tsmc18rf\n')
    if n_or_p == 'nmos':
        netlistOut.writelines('    cell_name: nmos2v\n')
    else:
        netlistOut.writelines('    cell_name: pmos2v\n')
    netlistOut.writelines('    instpins:\n')
    netlistOut.writelines('      B:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(b_port))
    netlistOut.writelines('        num_bits: 1\n')
    netlistOut.writelines('      D:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(d_port))
    netlistOut.writelines('        num_bits: 1\n')
    netlistOut.writelines('      G:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(g_port))
    netlistOut.writelines('        num_bits: 1\n')
    netlistOut.writelines('      S:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(s_port))
    netlistOut.writelines('        num_bits: 1\n')

    netlistOut.writelines('    master_prop:\n')
    netlistOut.writelines('      model:\n')
    if n_or_p == 'nmos':
        netlistOut.writelines('        defVal: nch\n')
    else:
        netlistOut.writelines('        defVal: pch\n')
    netlistOut.writelines('        prompt: Model name\n')
    netlistOut.writelines('      l:\n')
    netlistOut.writelines('        defVal: {}\n'.format(l))
    netlistOut.writelines('        prompt: l (M)\n')
    netlistOut.writelines('      w:\n')
    netlistOut.writelines('        defVal: {}\n'.format(w))
    netlistOut.writelines('        prompt: w (M)\n')
    netlistOut.writelines('      m:\n')
    netlistOut.writelines('        defVal: 1\n')
    netlistOut.writelines('        prompt: Multiplier\n')

    netlistOut.writelines('    inst_prop:\n')
    netlistOut.writelines('      l:\n')
    netlistOut.writelines('        value: {}\n'.format(l))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      m:\n')
    netlistOut.writelines('        value: 1\n')
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      fingers:\n')
    netlistOut.writelines('        value: {}\n'.format(m))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      w:\n')
    netlistOut.writelines('        value: {}\n'.format(w))
    netlistOut.writelines('        valueType: string\n')

    # netlistout.simp
    netlist_file.writelines('  {}:\n'.format(mos_name))
    if n_or_p == 'nmos':
        netlist_file.writelines('    cell_name: nmos2v\n')
    else:
        netlist_file.writelines('    cell_name: pmos2v\n')
    netlist_file.writelines('    instpins:\n')
    netlist_file.writelines('      B:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(b_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('      D:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(d_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('      G:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(g_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('      S:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(s_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('    lib_name: tsmc18rf\n')
    if n_or_p == 'nmos':
        netlist_file.writelines('    modelName: nch\n')
    else:
        netlist_file.writelines('    modelName: pch\n')
    netlist_file.writelines('    prop:\n')
    netlist_file.writelines('      l: {}\n'.format(l))
    netlist_file.writelines('      m: 1\n')
    netlist_file.writelines('      nf: {}\n'.format(m))
    netlist_file.writelines('      w: {}\n'.format(w))

    if tunable:
        param_dict.update({mos_name+'_w':w})
        #param_dict.update({mos_name+'_l':l})

def gen_c_out(netlistOut, netlist_file, ins_name, top_port, bot_port, value, unit):
    if (bot_port == '0') or (bot_port =='vss'):
        bot_port = 'GND'
    if (top_port == '0') or (top_port =='vss'):
        top_port = 'GND'
    if bot_port == 'vdd!':
        bot_port = 'VDD'
    if top_port == 'vdd!':
        top_port = 'VDD'
    l, w, c_new = get_cap_size(value, ins_name, top_port, bot_port)
    # netlistout file
    netlistOut.writelines('  {}:\n'.format(ins_name))
    netlistOut.writelines('    lib_name: tsmc18rf\n')
    netlistOut.writelines('    cell_name: mimcap\n')
    netlistOut.writelines('    instpins:\n')
    netlistOut.writelines('      BOT:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(bot_port))
    netlistOut.writelines('        num_bits: 1\n')
    netlistOut.writelines('      TOP:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(top_port))
    netlistOut.writelines('        num_bits: 1\n')

    netlistOut.writelines('    master_prop:\n')
    netlistOut.writelines('      model:\n')
    netlistOut.writelines('        defVal: mimcap\n')
    netlistOut.writelines('        prompt: Model name\n')
    netlistOut.writelines('      c:\n')
    netlistOut.writelines('        defVal: {}{}\n'.format(c_new,unit))
    netlistOut.writelines('        prompt: Capacitance(F)\n')
    netlistOut.writelines('      l:\n')
    netlistOut.writelines('        defVal: {}u\n'.format(l))
    netlistOut.writelines('        prompt: Length(M)\n')
    netlistOut.writelines('      w:\n')
    netlistOut.writelines('        defVal: {}u\n'.format(w))
    netlistOut.writelines('        prompt: Width(M)\n')

    netlistOut.writelines('    inst_prop:\n')
    netlistOut.writelines('      l:\n')
    netlistOut.writelines('        value: {}u\n'.format(l))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      m:\n')
    netlistOut.writelines('        value: 1\n')
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      w:\n')
    netlistOut.writelines('        value: {}u\n'.format(w))
    netlistOut.writelines('        valueType: string\n')

    # netlistout.simp file
    netlist_file.writelines('  {}:\n'.format(ins_name))
    netlist_file.writelines('    cell_name: mimcap\n')
    netlist_file.writelines('    instpins:\n')
    netlist_file.writelines('      BOT:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(bot_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('      TOP:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(top_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('    lib_name: tsmc18rf\n')
    netlist_file.writelines('    modelName: mimcap\n')
    netlist_file.writelines('    prop:\n')
    netlist_file.writelines('      l: {}u\n'.format(l))
    netlist_file.writelines('      m: 1\n')
    netlist_file.writelines('      w: {}u\n'.format(w))
    #return l, w, c_new

def gen_r_out(netlistOut, netlist_file, ins_name, top_port, bot_port, value, unit):
    if (bot_port == '0') or (bot_port =='vss'):
        bot_port = 'GND'
    if (top_port == '0') or (top_port =='vss'):
        top_port = 'GND'
    if bot_port == 'vdd!':
        bot_port = 'VDD'
    if top_port == 'vdd!':
        top_port = 'VDD'
    l, w, seg, r_new = get_res_size(value, ins_name)

    # netlistout file
    netlistOut.writelines('  {}:\n'.format(ins_name))
    netlistOut.writelines('    lib_name: tsmc18rf\n')
    netlistOut.writelines('    cell_name: rphripoly\n')
    netlistOut.writelines('    instpins:\n')
    netlistOut.writelines('      MINUS:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(bot_port))
    netlistOut.writelines('        num_bits: 1\n')
    netlistOut.writelines('      PLUS:\n')
    netlistOut.writelines('        direction: inputOutput\n')
    netlistOut.writelines('        net_name: "{}"\n'.format(top_port))
    netlistOut.writelines('        num_bits: 1\n')

    netlistOut.writelines('    master_prop:\n')
    netlistOut.writelines('      model:\n')
    netlistOut.writelines('        defVal: rppolyhri\n')
    netlistOut.writelines('        prompt: Model name\n')
    netlistOut.writelines('      segments:\n')
    netlistOut.writelines('        defVal: {}\n'.format(seg))
    netlistOut.writelines('        prompt: Number of segments\n')

    netlistOut.writelines('    inst_prop:\n')
    # netlistOut.writelines('      res:\n')
    # netlistOut.writelines('        value: {}{}\n'.format(value, unit))
    # netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      segments:\n')
    netlistOut.writelines('        value: {}\n'.format(seg))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      l:\n')
    netlistOut.writelines('        value: {}u\n'.format(l))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      sumL:\n')
    netlistOut.writelines('        value: {}u\n'.format(l*seg))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      w:\n')
    netlistOut.writelines('        value: {}u\n'.format(w))
    netlistOut.writelines('        valueType: string\n')
    netlistOut.writelines('      sumW:\n')
    netlistOut.writelines('        value: {}u\n'.format(w))
    netlistOut.writelines('        valueType: string\n')

    # netlistout.simp file
    netlist_file.writelines('  {}:\n'.format(ins_name))
    netlist_file.writelines('    cell_name: rphripoly\n')
    netlist_file.writelines('    instpins:\n')
    netlist_file.writelines('      MINUS:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(bot_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('      PLUS:\n')
    netlist_file.writelines('        direction: inputOutput\n')
    netlist_file.writelines('        net_name: {}\n'.format(top_port))
    netlist_file.writelines('        num_bits: 1\n')
    netlist_file.writelines('    lib_name: tsmc18rf\n')
    netlist_file.writelines('    modelName: rppolyhri\n')
    netlist_file.writelines('    prop:\n')
    netlist_file.writelines('      l: {}u\n'.format(l*seg))
    netlist_file.writelines('      m: 1\n')
    netlist_file.writelines('      w: {}u\n'.format(w))
    #return l, w, seg, r_new

def get_gm_negative_diff(netlist_file, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('subckt gm_negative_diff_{} vdd vin vip vo vss vb\n'.format(gm_index))
    ## M5
    netlist_file.writelines('    M5 (vb vb vss vss) nch l={l5} w={w5} m={m5} \\ \n'.format(l5=L5, w5=W5, m5=M5))
    netlist_file.writelines('        ad=({w5}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w5}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w5}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w5}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w5}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w5}*2) \n'.format(w5=W5))
    ## M4
    netlist_file.writelines('    M4 (net14 vb vss vss) nch l={l4} w={w4} m={m4} \\ \n'.format(l4=L4, w4=W4, m4=M4))
    netlist_file.writelines('        ad=({w4}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w4}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w4}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w4}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w4}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w4}*2) \n'.format(w4=W4))
    ## M1
    netlist_file.writelines('    M1 (vo vin net14 vss) nch l={l1} w={w1} m={m1} \\ \n'.format(l1=L1, w1=W1, m1=M1))
    netlist_file.writelines('        ad=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \n'.format(w1=W1))
    ## M0
    netlist_file.writelines('    M0 (net12 vip net14 vss) nch l={l0} w={w0} m={m0} \\ \n'.format(l0=L0, w0=W0, m0=M0))
    netlist_file.writelines('        ad=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \n'.format(w0=W0))
    ## M3
    netlist_file.writelines('    M3 (net12 net12 vdd vdd) pch l={l3} w={w3} m={m3} \\ \n'.format(l3=L3, w3=W3, m3=M3))
    netlist_file.writelines('        ad=({w3}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w3}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w3}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w3}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w3}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w3}*2) \n'.format(w3=W3))
    ## M2
    netlist_file.writelines('    M2 (vo net12 vdd vdd) pch l={l2} w={w2} m={m2} \\ \n'.format(l2=L2, w2=W2, m2=M2))
    netlist_file.writelines('        ad=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \n'.format(w2=W2))
    ## Idc
    #netlist_file.writelines('    I0 (vdd vb) isource dc=1u type=dc \n')
    ## END
    netlist_file.writelines('ends gm_negative_diff_{} \n'.format(gm_index))
    instance_name = 'gm_negative_diff_'+str(gm_index)
    return instance_name

def get_gm_negative_diff_lvs(netlist_file, gm_index, VDD, VIN, VIP, VO, VSS, VB, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('******************\n')
    netlist_file.writelines('* Library Name: OPAMP\n')
    netlist_file.writelines('* Cell Name: gm_negative_diff_{}\n'.format(gm_index))
    netlist_file.writelines('* View Name: schematic\n')
    
    # netlist_file.writelines('.SUBCKT gm_negative_diff_{} vdd vin vip vo vss\n'.format(gm_index))
    ## M5
    netlist_file.writelines('M5_{idx} {vb} {vb} {vss} {vss} N l={l5} w={w5} m={m5}\n'.format(idx=gm_index, vb=VB, vss=VSS, l5=L5, w5=W5, m5=M5))
    ## M4
    netlist_file.writelines('M4_{idx} NET14_{idx} {vb} {vss} {vss} N l={l4} w={w4} m={m4}\n'.format(idx=gm_index, vb=VB, vss=VSS, l4=L4, w4=W4, m4=M4))
    ## M1
    netlist_file.writelines('M1_{idx} {vo} {vin} NET14_{idx} {vss} N l={l1} w={w1} m={m1}\n'.format(idx=gm_index, vo=VO, vin=VIN, vss=VSS, l1=L1, w1=W1, m1=M1))
    ## M0
    netlist_file.writelines('M0_{idx} NET12_{idx} {vip} NET14_{idx} {vss} N l={l0} w={w0} m={m0}\n'.format(idx=gm_index, vip=VIP, vss=VSS, l0=L0, w0=W0, m0=M0))
    ## M3
    netlist_file.writelines('M3_{idx} NET12_{idx} NET12_{idx} {vdd} {vdd} P l={l3} w={w3} m={m3}\n'.format(idx=gm_index, vdd=VDD, l3=L3, w3=W3, m3=M3))
    ## M2
    netlist_file.writelines('M2_{idx} {vo} NET12_{idx} {vdd} {vdd} P l={l2} w={w2} m={m2}\n'.format(idx=gm_index, vo=VO, vdd=VDD, l2=L2, w2=W2, m2=M2))
    ## END
    # netlist_file.writelines('.ENDS\n')
    netlist_file.writelines('******************\n')
    instance_name = 'gm_negative_diff_'+str(gm_index)
    return instance_name

def gen_negative_diff_out(circuit_name, param_dict, M_index, S_index, manualcons_file, netlistOut, netlist_file, gm_index, vdd, vin, vip, vo, vss, vb, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    manualcons_file.writelines('M{} OPAMP_{}_M0_{} OPAMP_{}_M1_{} OPAMP_{}_M4_{}\n'.format(M_index, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M2_{} OPAMP_{}_M3_{}\n'.format(M_index+1, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M5_{} M{}\n'.format(M_index+2, circuit_name.upper(), gm_index, M_index+1))
    manualcons_file.writelines('S{} M{} M{}\n'.format(S_index, M_index, M_index+2))
    manualcons_file.writelines('\n')
    ## M5
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M5_'+str(gm_index), 'nmos', vb, vb, vss, vss, L5, W5, M5)
    ## M4
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M4_'+str(gm_index), 'nmos', 'NET14_'+str(gm_index), vb, vss, vss, L4, W4, M4)
    ## M1
    gen_mos_out(False, param_dict, netlistOut, netlist_file, 'M1_'+str(gm_index), 'nmos', vo, vin, 'NET14_'+str(gm_index), vss, L1, W1, M1)
    ## M0
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M0_'+str(gm_index), 'nmos', 'NET12_'+str(gm_index), vip, 'NET14_'+str(gm_index), vss, L0, W0, M0)
    ## M3
    gen_mos_out(False, param_dict, netlistOut, netlist_file, 'M3_'+str(gm_index), 'pmos', 'NET12_'+str(gm_index), 'NET12_'+str(gm_index), vdd, vdd, L3, W3, M3)
    ## M2
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M2_'+str(gm_index), 'pmos', vo, 'NET12_'+str(gm_index), vdd, vdd, L2, W2, M2)

def get_gm_negative_mid(netlist_file, gm_index, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('subckt gm_negative_mid_{} vdd vi vo vss vb\n'.format(gm_index))
    ## M4
    netlist_file.writelines('    M4 (vb vb vdd vdd) pch l={l4} w={w4} m={m4} \\ \n'.format(l4=L4, w4=W4, m4=M4))
    netlist_file.writelines('        ad=({w4}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w4}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w4}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w4}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w4}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w4}*2) \n'.format(w4=W4))
    ## M1
    netlist_file.writelines('    M1 (vo net7 vss vss) nch l={l1} w={w1} m={m1} \\ \n'.format(l1=L1, w1=W1, m1=M1))
    netlist_file.writelines('        ad=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \n'.format(w1=W1))
    ## M0
    netlist_file.writelines('    M0 (net7 net7 vss vss) nch l={l0} w={w0} m={m0} \\ \n'.format(l0=L0, w0=W0, m0=M0))
    netlist_file.writelines('        ad=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \n'.format(w0=W0))
    ## M3
    netlist_file.writelines('    M3 (vo vb vdd vdd) pch l={l3} w={w3} m={m3} \\ \n'.format(l3=L3, w3=W3, m3=M3))
    netlist_file.writelines('        ad=({w3}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w3}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w3}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w3}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w3}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w3}*2) \n'.format(w3=W3))
    ## M2
    netlist_file.writelines('    M2 (net7 vi vdd vdd) pch l={l2} w={w2} m={m2} \\ \n'.format(l2=L2, w2=W2, m2=M2))
    netlist_file.writelines('        ad=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \n'.format(w2=W2))
    ## Idc
    #netlist_file.writelines('    I0 (vdd vb) isource dc=1u type=dc \n')
    ## END
    netlist_file.writelines('ends gm_negative_mid_{} \n'.format(gm_index))
    instance_name = 'gm_negative_mid_'+str(gm_index)
    return instance_name

def get_gm_negative_mid_lvs(netlist_file, gm_index, VDD, VI, VO, VSS, VB, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('******************\n')
    netlist_file.writelines('* Library Name: OPAMP\n')
    netlist_file.writelines('* Cell Name: gm_negative_mid_{}\n'.format(gm_index))
    netlist_file.writelines('* View Name: schematic\n')
    
    # netlist_file.writelines('.SUBSKT gm_negative_mid_{} vdd vi vo vss \n'.format(gm_index))
    ## M4
    netlist_file.writelines('M4_{idx} {vb} {vb} {vdd} {vdd} P l={l4} w={w4} m={m4}\n'.format(idx=gm_index, vb=VB, vdd=VDD, l4=L4, w4=W4, m4=M4))
    ## M1
    netlist_file.writelines('M1_{idx} {vo} NET7_{idx} {vss} {vss} N l={l1} w={w1} m={m1}\n'.format(idx=gm_index, vo=VO, vss=VSS, l1=L1, w1=W1, m1=M1))
    ## M0
    netlist_file.writelines('M0_{idx} NET7_{idx} NET7_{idx} {vss} {vss} N l={l0} w={w0} m={m0}\n'.format(idx=gm_index, vss=VSS, l0=L0, w0=W0, m0=M0))
    ## M3
    netlist_file.writelines('M3_{idx} {vo} {vb} {vdd} {vdd} P l={l3} w={w3} m={m3}\n'.format(idx=gm_index, vb=VB, vo=VO, vdd=VDD, l3=L3, w3=W3, m3=M3))
    ## M2
    netlist_file.writelines('M2_{idx} NET7_{idx} {vi} {vdd} {vdd} P l={l2} w={w2} m={m2}\n'.format(idx=gm_index, vi=VI, vdd=VDD, l2=L2, w2=W2, m2=M2))
    ## END
    # netlist_file.writelines('.ENDS\n')
    netlist_file.writelines('******************\n')
    instance_name = 'gm_negative_mid_'+str(gm_index)
    return instance_name

def gen_negative_mid_out(circuit_name, param_dict, M_index, S_index, manualcons_file, netlistOut, netlist_file, gm_index, vdd, vi, vo, vss, vb, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    manualcons_file.writelines('M{} OPAMP_{}_M0_{} OPAMP_{}_M1_{}\n'.format(M_index, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M2_{} OPAMP_{}_M3_{}\n'.format(M_index+1, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M4_{} M{}\n'.format(M_index+2, circuit_name.upper(), gm_index, M_index+1))
    manualcons_file.writelines('S{} M{} M{}\n'.format(S_index, M_index, M_index+2))
    manualcons_file.writelines('\n')
    ## M4
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M4_'+str(gm_index), 'pmos', vb, vb, vdd, vdd, L4, W4, M4)
    ## M1
    gen_mos_out(False, param_dict, netlistOut, netlist_file, 'M1_'+str(gm_index), 'nmos', vo, 'NET7_'+str(gm_index), vss, vss, L1, W1, M1)
    ## M0
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M0_'+str(gm_index), 'nmos', 'NET7_'+str(gm_index), 'NET7_'+str(gm_index), vss, vss, L0, W0, M0)
    ## M3
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M3_'+str(gm_index), 'pmos', vo, vb, vdd, vdd, L3, W3, M3)
    ## M2
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M2_'+str(gm_index), 'pmos', 'NET7_'+str(gm_index), vi, vdd, vdd, L2, W2, M2)

def get_gm_positive_diff(netlist_file, gm_index, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('subckt gm_positive_diff_{} vdd vin vip vo vss vb\n'.format(gm_index))
    ## M5
    netlist_file.writelines('    M5 (vb vb vss vss) nch l={l5} w={w5} m={m5} \\ \n'.format(l5=L5, w5=W5, m5=M5))
    netlist_file.writelines('        ad=({w5}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w5}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w5}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w5}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w5}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w5}*2) \n'.format(w5=W5))
    ## M4
    netlist_file.writelines('    M4 (net4 vb vss vss) nch l={l4} w={w4} m={m4} \\ \n'.format(l4=L4, w4=W4, m4=M4))
    netlist_file.writelines('        ad=({w4}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w4}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w4}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w4}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w4}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w4}*2) \n'.format(w4=W4))
    ## M1
    netlist_file.writelines('    M1 (net5 vin net4 vss) nch l={l1} w={w1} m={m1} \\ \n'.format(l1=L1, w1=W1, m1=M1))
    netlist_file.writelines('        ad=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \n'.format(w1=W1))
    ## M0
    netlist_file.writelines('    M0 (vo vip net4 vss) nch l={l0} w={w0} m={m0} \\ \n'.format(l0=L0, w0=W0, m0=M0))
    netlist_file.writelines('        ad=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \n'.format(w0=W0))
    ## M3
    netlist_file.writelines('    M3 (net5 net5 vdd vdd) pch l={l3} w={w3} m={m3} \\ \n'.format(l3=L3, w3=W3, m3=M3))
    netlist_file.writelines('        ad=({w3}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w3}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w3}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w3}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w3}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w3}*2) \n'.format(w3=W3))
    ## M2
    netlist_file.writelines('    M2 (vo net5 vdd vdd) pch l={l2} w={w2} m={m2} \\ \n'.format(l2=L2, w2=W2, m2=M2))
    netlist_file.writelines('        ad=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \n'.format(w2=W2))
    ## Idc
    #netlist_file.writelines('    I0 (vdd vb) isource dc=1u type=dc \n')
    ## END
    netlist_file.writelines('ends gm_positive_diff_{} \n'.format(gm_index))
    instance_name = 'gm_positive_diff_'+str(gm_index)
    return instance_name

def get_gm_positive_diff_lvs(netlist_file, gm_index, VDD, VIN, VIP, VO, VSS, VB,  W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('******************\n')
    netlist_file.writelines('* Library Name: OPAMP\n')
    netlist_file.writelines('* Cell Name: gm_positive_diff_{}\n'.format(gm_index))
    netlist_file.writelines('* View Name: schematic\n')
    
    # netlist_file.writelines('.SUBCKT gm_positive_diff_{} vdd vin vip vo vss\n'.format(gm_index))
    ## M5
    netlist_file.writelines('M5_{idx} {vb} {vb} {vss} {vss} N l={l5} w={w5} m={m5}\n'.format(idx=gm_index, vb=VB, vss=VSS, l5=L5, w5=W5, m5=M5))
    ## M4
    netlist_file.writelines('M4_{idx} NET4_{idx} {vb} {vss} {vss} N l={l4} w={w4} m={m4}\n'.format(idx=gm_index, vb=VB, vss=VSS, l4=L4, w4=W4, m4=M4))
    ## M1
    netlist_file.writelines('M1_{idx} NET5_{idx} {vin} NET4_{idx} {vss} N l={l1} w={w1} m={m1}\n'.format(idx=gm_index, vin=VIN, vss=VSS, l1=L1, w1=W1, m1=M1))
    ## M0
    netlist_file.writelines('M0_{idx} {vo} {vip} NET4_{idx} {vss} N l={l0} w={w0} m={m0}\n'.format(idx=gm_index, vo=VO, vip=VIP, vss=VSS, l0=L0, w0=W0, m0=M0))
    ## M3
    netlist_file.writelines('M3_{idx} NET5_{idx} NET5_{idx} {vdd} {vdd} P l={l3} w={w3} m={m3}\n'.format(idx=gm_index, vdd=VDD, l3=L3, w3=W3, m3=M3))
    ## M2
    netlist_file.writelines('M2_{idx} {vo} NET5_{idx} {vdd} {vdd} P l={l2} w={w2} m={m2}\n'.format(idx=gm_index, vo=VO, vdd=VDD, l2=L2, w2=W2, m2=M2))
    ## END
    # netlist_file.writelines('.ENDS\n')
    netlist_file.writelines('******************\n')
    instance_name = 'gm_positive_diff_'+str(gm_index)
    return instance_name

def gen_positive_diff_out(circuit_name, param_dict, M_index, S_index, manualcons_file, netlistOut, netlist_file, gm_index, vdd, vin, vip, vo, vss, vb, W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    manualcons_file.writelines('M{} OPAMP_{}_M0_{} OPAMP_{}_M1_{} OPAMP_{}_M4_{}\n'.format(M_index, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M2_{} OPAMP_{}_M3_{}\n'.format(M_index+1, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M5_{} M{}\n'.format(M_index+2, circuit_name.upper(), gm_index, M_index+1))
    manualcons_file.writelines('S{} M{} M{}\n'.format(S_index, M_index, M_index+2))
    manualcons_file.writelines('\n')
    ## M5
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M5_'+str(gm_index), 'nmos', vb, vb, vss, vss, L5, W5, M5)
    ## M4
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M4_'+str(gm_index), 'nmos', 'NET4_'+str(gm_index), vb, vss, vss, L4, W4, M4)
    ## M1
    gen_mos_out(False, param_dict, netlistOut, netlist_file, 'M1_'+str(gm_index), 'nmos', 'NET5_'+str(gm_index), vin, 'NET4_'+str(gm_index), vss, L1, W1, M1)
    ## M0
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M0_'+str(gm_index), 'nmos', vo, vip, 'NET4_'+str(gm_index), vss, L0, W0, M0)
    ## M3
    gen_mos_out(False, param_dict, netlistOut, netlist_file, 'M3_'+str(gm_index), 'pmos', 'NET5_'+str(gm_index), 'NET5_'+str(gm_index), vdd, vdd, L3, W3, M3)
    ## M2
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M2_'+str(gm_index), 'pmos', vo, 'NET5_'+str(gm_index), vdd, vdd, L2, W2, M2)

def get_gm_positive_mid(netlist_file, gm_index, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('subckt gm_positive_mid_{} vdd vi vo vss vb\n'.format(gm_index))
    ## M2
    netlist_file.writelines('    M2 (vb vb vss vss) nch l={l2} w={w2} m={m2} \\ \n'.format(l2=L2, w2=W2, m2=M2))
    netlist_file.writelines('        ad=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w2}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w2}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w2}*2) \n'.format(w2=W2))
    ## M0
    netlist_file.writelines('    M0 (vo vb vss vss) nch l={l0} w={w0} m={m0} \\ \n'.format(l0=L0, w0=W0, m0=M0))
    netlist_file.writelines('        ad=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w0}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w0}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w0}*2) \n'.format(w0=W0))
    ## M1
    netlist_file.writelines('    M1 (vo vi vdd vdd) pch l={l1} w={w1} m={m1} \\ \n'.format(l1=L1, w1=W1, m1=M1))
    netlist_file.writelines('        ad=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*int(1/2))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        as=({w1}*((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))+((0*0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2))))/1 \\ \n        pd=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(1/2)*5.4e-07*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*((1+1)*(int((1+1)/2)-int(1/2))+(1+0)*(int((1+2)/2)-int((1+1)/2))))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*int(1/2))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        ps=(((4.8e-07+((1-1)*5.4e-07)/2+0)*(int((1+1)/2)-int(1/2))+(4.8e-07+4.8e-07+(int(1/2)-1)*5.4e-07+0+0)*(int((1+2)/2)-int((1+1)/2)))*2+(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-((0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0))*(int((1+1)/2)-int(1/2))+(0*(((1+1)*(int((1+1)/2)-int(1/2))+(1+2)*(int((1+2)/2)-int((1+1)/2)))-0-0))*(int((1+2)/2)-int((1+1)/2))))*{w1}+((0*((1-1)/2+1-0))*(int((1+1)/2)-int(1/2))+(0*(int(1/2)+1-0-0))*(int((1+2)/2)-int((1+1)/2)))*4)/1 \\ \n        nrd=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \\ \n        nrs=(5.4e-07+2.7e-07*(1*2-2))/1/({w1}*2) \n'.format(w1=W1))
    ## Idc
    #netlist_file.writelines('    I0 (vdd vb) isource dc=1u type=dc \n')
    ## END
    netlist_file.writelines('ends gm_positive_mid_{} \n'.format(gm_index))
    instance_name = 'gm_positive_mid_'+str(gm_index)
    return instance_name

def get_gm_positive_mid_lvs(netlist_file, gm_index, VDD, VI, VO, VSS, VB, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    netlist_file.writelines('******************\n')
    netlist_file.writelines('* Library Name: OPAMP\n')
    netlist_file.writelines('* Cell Name: gm_positive_mid_{}\n'.format(gm_index))
    netlist_file.writelines('* View Name: schematic\n')
    
    # netlist_file.writelines('.SUBCKT gm_positive_mid_{} vdd vi vo vss\n'.format(gm_index))
    ## M2
    netlist_file.writelines('M2_{idx} {vb} {vb} {vss} {vss} N l={l2} w={w2} m={m2}\n'.format(idx=gm_index, vb=VB, vss=VSS, l2=L2, w2=W2, m2=M2))
    ## M0
    netlist_file.writelines('M0_{idx} {vo} {vb} {vss} {vss} N l={l0} w={w0} m={m0}\n'.format(idx=gm_index, vo=VO, vb=VB, vss=VSS, l0=L0, w0=W0, m0=M0))
    ## M1
    netlist_file.writelines('M1_{idx} {vo} {vi} {vdd} {vdd} P l={l1} w={w1} m={m1}\n'.format(idx=gm_index, vo=VO, vi=VI, vdd=VDD, l1=L1, w1=W1, m1=M1))
    ## END
    # netlist_file.writelines('.ENDS\n')
    netlist_file.writelines('******************\n')
    instance_name = 'gm_positive_mid_'+str(gm_index)
    return instance_name

def gen_positive_mid_out(circuit_name, param_dict, M_index, S_index, manualcons_file, netlistOut, netlist_file, gm_index, vdd, vi, vo, vss, vb, W2, L2, M2, W1, L1, M1, W0, L0, M0):
    # manualcons_file.writelines('S{} opamp_{}_M0_{} opamp_{}_M1_{}\n'.format(S_index, circuit_name, gm_index, circuit_name, gm_index))
    # manualcons_file.writelines('P{} S{} opamp_{}_M2_{}\n'.format(P_index, S_index, circuit_name, gm_index))
    manualcons_file.writelines('M{} OPAMP_{}_M1_{} OPAMP_{}_M2_{}\n'.format(M_index, circuit_name.upper(), gm_index, circuit_name.upper(), gm_index))
    manualcons_file.writelines('S{} OPAMP_{}_M0_{} M{}\n'.format(S_index, circuit_name.upper(), gm_index, M_index))
    manualcons_file.writelines('\n')
    ## M2
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M2_'+str(gm_index), 'nmos', vb, vb, vss, vss, L2, W2, M2)
    ## M0
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M0_'+str(gm_index), 'nmos', vo, vb, vss, vss, L0, W0, M0)
    ## M1
    gen_mos_out(True, param_dict, netlistOut, netlist_file, 'M1_'+str(gm_index), 'pmos', vo, vi, vdd, vdd, L1, W1, M1)

# cal W/L/vb of transistors in each gm block with gmid design method
# detail gms' topology plz refer to my document

def revise_WL(original_w, original_l, wmax, wmin, lmax): # revise the calculated w/l to [min, max]
    # um -> m
    #original_l = original_l*1e-6
    original_w = original_w*1e-6

    if (wmin<=original_w<=wmax):
        return round(original_w,8), round(original_l,8), 1
    elif original_w>wmax:
        m = math.ceil(original_w/10)
        if (m%2==0):
            w = original_w / m
        else:
            m += 1
            w = original_w / m
        return round(w,8), round(original_l,8), m
    else:
        t = wmin / original_w
        w = wmin
        l = original_l * t
        l = min(lmax, l)
        return round(w,8), round(l,8), 1

def revise_proportional_wl(l, w):
    w = w*1e-6
    if w > wmax:
        m = math.ceil(w/10)
        if (m%2==0):
            w = w / m
        else:
            m += 1
            w = w / m
    elif w < wmin:
        l = l * (wmin/w)
        l = min(lmax, l)
        w = wmin
        m = 1
    else:
        m = 1
    return l, w, m

def find_l_in_list(l):
    idx = (np.abs(L_array - l)).argmin()
    return L_array[idx]

def cal_single_mos(intrinsic_gain_times, gmid_target, gm_cal, gds_cal, gm_lut, gds_lut, id_lut, debug=None):
    if debug == None:
        debug = False
    # TODO the efficiency of table lookup here can be further improved
    vgs_list       = gm_lut.index.values.tolist()
    intrinsic_gain = gm_cal/gds_cal
    lu_method = 1
    
    # search for gm/id = gmid_target, empirically 15 S/A (nmos) or 10 S/A (pmos) and gm/gds = intrinsic_gain_times * (gm_cal/gds_cal) best
    # gmid_target can be further optimized, which is equivalent to directly optimizing W/L

    ## method1, incomplete traversal, but fast
    if lu_method == 1:
        if debug:
            print('[Debug] using method1 for LUT search, faster!')
        best_l = 1.00311557788945e-06
        #best_l = 5.00100502512563e-07
        best_vgs = vgs_list[0]
        for vgs in vgs_list:
            gm  = gm_lut.loc[vgs][best_l]
            id  = id_lut.loc[vgs][best_l]
            if abs(abs(gm/id)-gmid_target) <  abs(abs(gm_lut.loc[best_vgs][best_l]/id_lut.loc[best_vgs][best_l]) - gmid_target):
                best_vgs = vgs

        for l in L_list:
            gm  = gm_lut.loc[best_vgs][l]
            gds = gds_lut.loc[best_vgs][l]
            if abs(abs(gm/gds) - intrinsic_gain_times*intrinsic_gain) < abs(abs(gm_lut.loc[best_vgs][best_l]/gds_lut.loc[best_vgs][best_l]) -intrinsic_gain_times*intrinsic_gain):
                best_l   = l
    ## method2, complete traversal, slower
    else:
        if debug:
            print('[Debug] using method2 for LUT search, slower...')
        # 1.search for gm/id = gmid_target, empirically 15 S/A (nmos) or 10 S/A (pmos)
        # gmid_target can be further optimized, which is equivalent to directly optimizing W/L
        targetSA_list = []
        for vgs in vgs_list:
            #targetSA_L = L_list[0]
            for L in L_list:
                gm  = gm_lut.loc[vgs][L]
                id  = id_lut.loc[vgs][L]
                # if abs(abs(gm/id)-gmid_target) <  abs(abs(gm_lut.loc[vgs][targetSA_L]/id_lut.loc[vgs][targetSA_L]) - gmid_target):
                #     targetSA_L = L
                if abs(abs(gm_lut.loc[vgs][L]/id_lut.loc[vgs][L]) - gmid_target) < 1:
                    targetSA = [vgs, L]
                    targetSA_list.append(targetSA)
        # 2. search for gm/gds = 1.5 * (gm_cal/gds_cal) best
        best_vgs       = targetSA_list[0][0]
        best_l         = targetSA_list[0][1]
        for targetSA in targetSA_list:
            vgs = targetSA[0]
            l   = targetSA[1]
            gm  = gm_lut.loc[vgs][l]
            gds = gds_lut.loc[vgs][l]
            if abs(abs(gm/gds) - intrinsic_gain_times*intrinsic_gain) < abs(abs(gm_lut.loc[best_vgs][best_l]/gds_lut.loc[best_vgs][best_l]) -intrinsic_gain_times*intrinsic_gain):
                best_vgs = vgs
                best_l   = l
    if debug:
        print('[Debug] intrinsic_gain_times*intrinsic_gain = {}'.format(intrinsic_gain_times*intrinsic_gain))
        print('[Debug] the searched gm = {}'.format(gm_lut.loc[best_vgs][best_l]))
        print('[Debug] the searched gm*ro = {}'.format(gm_lut.loc[best_vgs][best_l]/gds_lut.loc[best_vgs][best_l]))
        print('[Debug] the searched gm/id = {}'.format(gm_lut.loc[best_vgs][best_l]/id_lut.loc[best_vgs][best_l]))

    # get the W, L and Id
    final_W  = gm_cal / gm_lut.loc[best_vgs][best_l] * 1  # um, because we fix the W = 1um when scanning the lut
    final_Id = id_lut.loc[best_vgs][best_l] * final_W # current relative to W=1um mos's current
    final_L  = best_l # m, l in l_list is 'm'
    #final_W, final_L, final_M = revise_WL(final_W, final_L, wmax, wmin, lmax)

    return final_W, final_L, final_Id

def cal_positive_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=None, gmid_target_1=None, gmid_target_2=None):
    if debug == None:
        debug = False

    # gm_cal and gds_cal are from topology optimization results
    if gmid_target_1==None:
        gmid_target_1 = 15
    if gmid_target_2==None:
        gmid_target_2 = 5
    
    if debug:
        print('[Debug] gmid_target_1: {}, gmid_target_2: {}'.format(gmid_target_1, gmid_target_2))

    W0, L0, Id0 = cal_single_mos(intrinsic_gain_times, gmid_target_1, gm_cal, gds_cal, gm_n, gds_n, id_n, debug=debug)
    W1, L1 = W0, L0
    L2, L3 = 1*L0, 1*L1  # 1x or 1.5x

    # get pmos's W based on pmos's lut
    # for pmos here, search for the gm that make the gm/id (gmid_target) = 8 S/A best
    vgs_list_p = gm_p.index.values.tolist()
    best_vgs_p = vgs_list_p[0]
    for vgs_p in vgs_list_p:
        gm = gm_p.loc[vgs_p][L2]
        id = id_p.loc[vgs_p][L2]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_p.loc[best_vgs_p][L2] / id_p.loc[best_vgs_p][L2]) - gmid_target_2):
            best_vgs_p = vgs_p
    idp = id_p.loc[best_vgs_p][L2]
    W2 = abs(Id0/idp) # um, because we fix the W = 1um when scanning the lut
    W3 = W2
    
    # get the tail mos's size and bias nmos size
    # note that the bias voltage is actually determined by the current mirro circuit 
    # tail nmos
    L4 = L0
    W4 = 2*W0
    # mirro nmos
    L5 = L4
    W5 = abs(W4/(2*Id0)*1e-6) # um, because the we set the Idc=1uA in the circuit

    if debug:
        print('[Debug] before revise:\n W5:{}\n L5:{}\n W4:{}\n L4:{}\n W3:{}\n L3:{}\n W2:{}\n L2:{}\n W1:{}\n L1:{}\n W0:{}\n L0:{}'.format(W5*1e-6, L5, W4*1e-6, L4, W3*1e-6, L3, W2*1e-6, L2, W1*1e-6, L1, W0*1e-6, L0))
    
    '''revise W/L to PDK supported bound'''
    W0, L0, M0 = revise_WL(W0, L0, wmax, wmin, lmax)
    l_tmp = L0
    L0 = find_l_in_list(L0)
    if debug:
        print('[Debug] switch L0 from {} to {}'.format(l_tmp, L0))
    W1, L1, M1 = W0, L0, M0 # um
    L2 = L0
    for vgs_p in vgs_list_p:
        gm = gm_p.loc[vgs_p][L2]
        id = id_p.loc[vgs_p][L2]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_p.loc[best_vgs_p][L2] / id_p.loc[best_vgs_p][L2]) - gmid_target_2):
            best_vgs_p = vgs_p
    idp = id_p.loc[best_vgs_p][L2]
    W2 = abs(Id0/idp) # um, because we fix the W = 1um when scanning the lut
    #print('[fuck] W2={}, L2={}'.format(W2, L2))
    L2, W2, M2 = revise_proportional_wl(L2, W2)
    L3, W3, M3 = L2, W2, M2
    L4 = L0
    W4 = 2*W0
    #print('[fuck] W4={}, L4={}'.format(W4, L4))
    L4, W4, M4 = revise_proportional_wl(L4, W4*1e6)
    L5 = L4
    W5 = abs(W4/(2*Id0)*1e-6) # m, because the we set the Idc=1uA in the circuit
    #print('[fuck] W5={}, L5={}'.format(W5, L5))
    L5, W5, M5 = revise_proportional_wl(L5, W5*1e6)

    if debug:
        print('[Debug] after revise:\n W5:{}\n L5:{}\n M5:{}\n W4:{}\n L4:{}\n M4:{}\n W3:{}\n L3:{}\n M3:{}\n W2:{}\n L2:{}\n M2:{}\n W1:{}\n L1:{}\n M1:{}\n W0:{}\n L0:{}\n M0:{}'.format(W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0))

    return W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0

def cal_positive_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=None, gmid_target_1=None, gmid_target_2=None):
    if debug == None:
        debug = False

    # gm_cal and gds_cal are from topology optimization results
    if gmid_target_1==None:
        gmid_target_1 = 10
    if gmid_target_2==None:
        gmid_target_2 = 8

    if debug:
        print('[Debug] gmid_target_1: {}, gmid_target_2: {}'.format(gmid_target_1, gmid_target_2))

    W1, L1, Id1 = cal_single_mos(intrinsic_gain_times, gmid_target_1, gm_cal, gds_cal, gm_p, gds_p, id_p, debug=debug)

    # get nmos's size
    # for nmos here, search for the gm that make the gm/id (gmid_target) = 8 S/A best
    L0 = L1
    vgs_list_n = gm_n.index.values.tolist()
    best_vgs_n = vgs_list_n[0]
    for vgs_n in vgs_list_n:
        gm = gm_n.loc[vgs_n][L0]
        id = id_n.loc[vgs_n][L0]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_n.loc[best_vgs_n][L0] / id_n.loc[best_vgs_n][L0]) - gmid_target_2):
            best_vgs_n = vgs_n
    idn = id_n.loc[best_vgs_n][L0]
    W0  = abs(Id1/idn) # um, because we fix the W = 1um when scanning the lut

    # mirro mos
    L2 = L0
    W2 = abs(W0 / Id1 * 1e-6) # um, because the we set the Idc=1uA in the circuit

    if debug:
        print('[Debug] before revise:\n W2:{}\n L2:{}\n W1:{}\n L1:{}\n W0:{}\n L0:{}\n idn:{}'.format(W2*1e-6, L2, W1*1e-6, L1, W0*1e-6, L0, idn))

    '''revise W/L to PDK supported bound'''
    W1, L1, M1 = revise_WL(W1, L1, wmax, wmin, lmax)
    l_tmp = L1
    L1 = find_l_in_list(L1)
    if debug:
        print('[Debug] switch L1 from {} to {}'.format(l_tmp, L1))
    L0 = L1
    for vgs_n in vgs_list_n:
        gm = gm_n.loc[vgs_n][L0]
        id = id_n.loc[vgs_n][L0]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_n.loc[best_vgs_n][L0] / id_n.loc[best_vgs_n][L0]) - gmid_target_2):
            best_vgs_n = vgs_n
    idn = id_n.loc[best_vgs_n][L0]
    W0  = abs(Id1/idn) # um, because we fix the W = 1um when scanning the lut
    #print('[fuck] W0={}, L0={}'.format(W0, L0))
    L0, W0, M0 = revise_proportional_wl(L0, W0) # m
    L2 = L0
    W2 = abs(W0 / Id1) # m, because the we set the Idc=1uA in the circuit
    #print('[fuck] W2={}, L2={}'.format(W2, L2))
    L2, W2, M2 = revise_proportional_wl(L2, W2)

    if debug:
        print('[Debug] after revise:\n W2:{}\n L2:{}\n M2:{}\n W1:{}\n L1:{}\n M1:{}\n W0:{}\n L0:{}\n M0:{}\n idn:{}'.format(W2, L2, M2, W1, L1, M1, W0, L0, M0, idn))
    
    return W2, L2, M2, W1, L1, M1, W0, L0, M0

def cal_negative_diff(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=None, gmid_target_1=None, gmid_target_2=None):
    if debug == None:
        debug = False

    # exactly same as cal_positive_diff
    if gmid_target_1==None:
        gmid_target_1 = 10
    if gmid_target_2==None:
        gmid_target_2 = 8

    if debug:
        print('[Debug] gmid_target_1: {}, gmid_target_2: {}'.format(gmid_target_1, gmid_target_2))

    W0, L0, Id0 = cal_single_mos(intrinsic_gain_times, gmid_target_1, gm_cal, gds_cal, gm_n, gds_n, id_n, debug=debug)
    W1, L1 = W0, L0
    L2, L3 = 1*L0, 1*L1  # 1x or 1.5x

    # get pmos's W based on pmos's lut
    # for pmos here, search for the gm that make the gm/id (gmid_target) = 8 S/A best
    vgs_list_p = gm_p.index.values.tolist()
    best_vgs_p = vgs_list_p[0]
    for vgs_p in vgs_list_p:
        gm = gm_p.loc[vgs_p][L2]
        id = id_p.loc[vgs_p][L2]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_p.loc[best_vgs_p][L2] / id_p.loc[best_vgs_p][L2]) - gmid_target_2):
            best_vgs_p = vgs_p
    idp = id_p.loc[best_vgs_p][L2]
    W2 = abs(Id0/idp) # um, because we fix the W = 1um when scanning the lut
    W3 = W2

    # get the tail mos's size and bias nmos size
    # note that the bias voltage is actually determined by the current mirro circuit 
    # tail nmos
    L4 = L0
    W4 = 2*W0
    # mirro nmos
    L5 = L4
    W5 = abs(W4/(2*Id0)*1e-6) # um, because the we set the Idc=1uA in the circuit

    if debug:
        print('[Debug] before revise:\n W5:{}\n L5:{}\n W4:{}\n L4:{}\n W3:{}\n L3:{}\n W2:{}\n L2:{}\n W1:{}\n L1:{}\n W0:{}\n L0:{}'.format(W5*1e-6, L5, W4*1e-6, L4, W3*1e-6, L3, W2*1e-6, L2, W1*1e-6, L1, W0*1e-6, L0))

    '''revise W/L to PDK supported bound'''
    W0, L0, M0 = revise_WL(W0, L0, wmax, wmin, lmax)
    l_tmp = L0
    L0 = find_l_in_list(L0)
    if debug:
        print('[Debug] switch L0 from {} to {}'.format(l_tmp, L0))
    W1, L1, M1 = W0, L0, M0
    L2 = L1
    for vgs_p in vgs_list_p:
        gm = gm_p.loc[vgs_p][L2]
        id = id_p.loc[vgs_p][L2]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_p.loc[best_vgs_p][L2] / id_p.loc[best_vgs_p][L2]) - gmid_target_2):
            best_vgs_p = vgs_p
    idp = id_p.loc[best_vgs_p][L2]
    W2 = abs(Id0/idp) # um, because we fix the W = 1um when scanning the lut
    L2, W2, M2 = revise_proportional_wl(L2, W2)
    L3, W3, M3 = L2, W2, M2
    L4 = L0
    W4 = 2*W0
    L4, W4, M4 = revise_proportional_wl(L4, W4*1e6)
    L5 = L4
    W5 = abs(W4/(2*Id0)*1e-6) # um, because the we set the Idc=1uA in the circuit
    L5, W5, M5 = revise_proportional_wl(L5, W5*1e6)

    if debug:
        print('[Debug] after revise:\n W5:{}\n L5:{}\n M5:{}\n W4:{}\n L4:{}\n M4:{}\n W3:{}\n L3:{}\n M3:{}\n W2:{}\n L2:{}\n M2:{}\n W1:{}\n L1:{}\n M1:{}\n W0:{}\n L0:{}\n M0:{}'.format(W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0))

    return W5, L5, M5, W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0

def cal_negative_mid(intrinsic_gain_times, gm_cal, gds_cal, gm_n, gds_n, id_n, gm_p, gds_p, id_p, debug=None, gmid_target_1=None, gmid_target_2=None):
    if debug == None:
        debug = False

    # gm_cal and gds_cal are from topology optimization results
    if gmid_target_1==None:
        gmid_target_1 = 10
    if gmid_target_2==None:
        gmid_target_2 = 6

    if debug:
        print('[Debug] gmid_target_1: {}, gmid_target_2: {}'.format(gmid_target_1, gmid_target_2))

    W2, L2, Id2 = cal_single_mos(intrinsic_gain_times, gmid_target_1, gm_cal, gds_cal, gm_p, gds_p, id_p, debug=debug)

    # get nmos's size
    # for nmos here, search for the gm that make the gm/id (gmid_target) = 8 S/A best
    L0 = L2
    vgs_list_n = gm_n.index.values.tolist()
    best_vgs_n = vgs_list_n[0]
    for vgs_n in vgs_list_n:
        gm = gm_n.loc[vgs_n][L0]
        id = id_n.loc[vgs_n][L0]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_n.loc[best_vgs_n][L0] / id_n.loc[best_vgs_n][L0]) - gmid_target_2):
            best_vgs_n = vgs_n
    idn = id_n.loc[best_vgs_n][L0]
    W0  = abs(Id2/idn) * 1 # um, because we fix the W = 1um when scanning the lut
    W1, L1 = W0, L0

    # get tail mos size
    # for pmos here, search for the gm that make the gm/id (gmid_target) = 6 S/A best
    L3 = L1
    vgs_list_p = gm_p.index.values.tolist()
    best_vgs_p = vgs_list_p[0]
    for vgs_p in vgs_list_p:
        gm = gm_p.loc[vgs_p][L3]
        id = id_p.loc[vgs_p][L3]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_p.loc[best_vgs_p][L3] / id_p.loc[best_vgs_p][L3]) - gmid_target_2):
            best_vgs_p = vgs_p
    idp = id_p.loc[best_vgs_p][L3]
    W3  = abs(Id2/idp) * 1 # um, because we fix the W = 1um when scanning the lut

    # mirro mos
    L4 = L3
    W4 = abs(W3 / Id2 * 1e-6) # um, because the we set the Idc=1uA in the circuit

    if debug:
        print('[Debug] before revise:\n W4:{}\n L4:{}\n W3:{}\n L3:{}\n W2:{}\n L2:{}\n W1:{}\n L1:{}\n W0:{}\n L0:{}'.format(W4*1e-6, L4, W3*1e-6, L3, W2*1e-6, L2, W1*1e-6, L1, W0*1e-6, L0))
    
    '''revise W/L to PDK supported bound'''
    W2, L2, M2 = revise_WL(W2, L2, wmax, wmin, lmax)
    l_tmp = L2
    L2 = find_l_in_list(L2)
    if debug:
        print('[Debug] switch L2 from {} to {}'.format(l_tmp, L2))
    W3, L3, M3 = W2, L2, M2
    L0 = L2
    for vgs_n in vgs_list_n:
        gm = gm_n.loc[vgs_n][L0]
        id = id_n.loc[vgs_n][L0]
        if abs(abs(gm/id) - gmid_target_2) < abs(abs(gm_n.loc[best_vgs_n][L0] / id_n.loc[best_vgs_n][L0]) - gmid_target_2):
            best_vgs_n = vgs_n
    idn = id_n.loc[best_vgs_n][L0]
    W0  = abs(Id2/idn) * 1 # um, because we fix the W = 1um when scanning the lut
    #print('[fuck] W0={}, L0={}'.format(W0, L0))
    L0, W0, M0 = revise_proportional_wl(L0, W0)
    L1, W1, M1 = L0, W0, M0
    L4 = L3
    W4 = abs(W3 / Id2) # um, because the we set the Idc=1uA in the circuit
    #print('[fuck] W4={}, L4={}'.format(W4, L4))
    L4, W4, M4 = revise_proportional_wl(L4, W4)

    if debug:
        print('[Debug] after revise:\n W4:{}\n L4:{}\n M4:{}\n W3:{}\n L3:{}\n M3:{}\n W2:{}\n L2:{}\n M2:{}\n W1:{}\n L1:{}\n M1:{}\n W0:{}\n L0:{}\n M0:{}'.format(W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0))

    return W4, L4, M4, W3, L3, M3, W2, L2, M2, W1, L1, M1, W0, L0, M0

