# translate a DAG into a spectre netlist file 

import os
import sys
import string
import random
import numpy as np

def get_node_info(x):

    if x == 0:
        instance_head = []
        node          = []
        instance      = []
        instance_val = []
        unit          = []
        bound         = []
        additional_node = []
        additional_ins_head  = []
        additional_ins  = []
        additional_ins_val  = []
        additional_ins_unit  = []
        additional_ins_bound  = []
    elif x == 1: # R
        instance_head = ['R']
        node          = [0,1]
        instance      = ['resistor']
        instance_val = ['r']
        unit          = ['*1M']
        bound         = ['1']
        additional_node = []
        additional_ins_head  = []
        additional_ins  = []
        additional_ins_val  = []
        additional_ins_unit  = []
        additional_ins_bound  = []
    elif x == 2: # C
        instance_head = ['C']
        node          = [0,1]
        instance      = ['capacitor']
        instance_val = ['c']
        unit          = ['*1p']
        bound         = ['10']
        additional_node = []
        additional_ins_head  = []
        additional_ins  = []
        additional_ins_val  = []
        additional_ins_unit  = []
        additional_ins_bound  = []
    elif x == 3: # RC parallel
        instance_head = ['R','C']
        node          = [0,1]
        instance      = ['resistor','capacitor']
        instance_val = ['r','c']
        unit          = ['*1M','*1p']
        bound         = ['1','10']
        additional_node = []
        additional_ins_head  = []
        additional_ins  = []
        additional_ins_val  = []
        additional_ins_unit  = []
        additional_ins_bound  = []
    elif x == 4: # RC series
        instance_head = ['R','C']
        node          = [0,1]
        instance      = ['resistor','capacitor']
        instance_val = ['r','c']
        unit          = ['*1M','*1p']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = []
        additional_ins  = []
        additional_ins_val  = []
        additional_ins_unit  = []
        additional_ins_bound  = []
    elif x == 5: # Feedforward +gm
        instance_head = ['G']
        node          = [1,0]
        instance      = ['vccs']
        instance_val = ['gm']
        unit          = ['*1m']
        bound         = ['10']
        additional_node = [1]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 6: # Feedforward -gm
        instance_head = ['G']
        node          = [1,0]
        instance      = ['vccs']
        instance_val = ['gm']
        unit          = ['*-1m']
        bound         = ['10']
        additional_node = [1]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
###
    elif x == 7: # R&5 series
        instance_head = ['G','R']
        node          = [0,1]
        instance      = ['vccs','resistor']
        instance_val = ['gm','r']
        unit          = ['*1m','*1M']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 8: # C&5 series
        instance_head = ['G','C']
        node          = [0,1]
        instance      = ['vccs','capacitor']
        instance_val = ['gm','c']
        unit          = ['*1m','*1p']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 9: # R&6 series
        instance_head = ['G','R']
        node          = [0,1]
        instance      = ['vccs','resistor']
        instance_val = ['gm','r']
        unit          = ['*-1m','*1M']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 10: # C&6 series
        instance_head = ['G','C']
        node          = [0,1]
        instance      = ['vccs','capacitor']
        instance_val = ['gm','c']
        unit          = ['*-1m','*1p']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 11: # R&5 parallel
        instance_head = ['G','R']
        node          = [1,0]
        instance      = ['vccs','resistor']
        instance_val = ['gm','r']
        unit          = ['*1m','*1M']
        bound         = ['10','10']
        additional_node = [1]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 12: # C&5 parallel
        instance_head = ['G','C']
        node          = [1,0]
        instance      = ['vccs','capacitor']
        instance_val = ['gm','c']
        unit          = ['*1m','*1p']
        bound         = ['10','10']
        additional_node = [1]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 13: # R&6 parallel
        instance_head = ['G','R']
        node          = [1,0]
        instance      = ['vccs','resistor']
        instance_val = ['gm','r']
        unit          = ['*-1m','*1M']
        bound         = ['10','10']
        additional_node = [1]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 14: # C&6 parallel
        instance_head = ['G','C']
        node          = [1,0]
        instance      = ['vccs','capacitor']
        instance_val = ['gm','c']
        unit          = ['*-1m','*1p']
        bound         = ['10','10']
        additional_node = [1]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
###
    elif x == 15: # Feedback +gm
        instance_head = ['G']
        node          = [0,1]
        instance      = ['vccs']
        instance_val = ['gm']
        unit          = ['*1m']
        bound         = ['10']
        additional_node = [0]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 16: # Feedback -gm
        instance_head = ['G']
        node          = [0,1]
        instance      = ['vccs']
        instance_val = ['gm']
        unit          = ['*-1m']
        bound         = ['10']
        additional_node = [0]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 17: # R&15 parallel
        instance_head = ['R','G']
        node          = [0,1]
        instance      = ['resistor','vccs']
        instance_val = ['r','gm']
        unit          = ['*1M','*1m']
        bound         = ['10','10']
        additional_node = [0]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 18: # R&15 series
        instance_head = ['R','G']
        node          = [0,1]
        instance      = ['resistor','vccs']
        instance_val = ['r','gm']
        unit          = ['*1M','*1m']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 19: # C&15 parallel
        instance_head = ['C','G']
        node          = [0,1]
        instance      = ['capacitor','vccs']
        instance_val = ['c','gm']
        unit          = ['*1p','*1m']
        bound         = ['10','10']
        additional_node = [0]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 20: # C&15 series
        instance_head = ['C','G']
        node          = [0,1]
        instance      = ['capacitor','vccs']
        instance_val = ['c','gm']
        unit          = ['*1p','*1m']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 21: # R&16 parallel
        instance_head = ['R','G']
        node          = [0,1]
        instance      = ['resistor','vccs']
        instance_val = ['r','gm']
        unit          = ['*1M','*-1m']
        bound         = ['10','10']
        additional_node = [0]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 22: # R&16 series
        instance_head = ['R','G']
        node          = [0,1]
        instance      = ['resistor','vccs']
        instance_val = ['r','gm']
        unit          = ['*1M','*-1m']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 23: # C&16 parallel
        instance_head = ['C','G']
        node          = [0,1]
        instance      = ['capacitor','vccs']
        instance_val = ['c','gm']
        unit          = ['*1p','*-1m']
        bound         = ['10','10']
        additional_node = [0]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    elif x == 24: # C&16 series
        instance_head = ['C','G']
        node          = [0,1]
        instance      = ['capacitor','vccs']
        instance_val = ['c','gm']
        unit          = ['*1p','*-1m']
        bound         = ['10','10']
        additional_node = [2]
        additional_ins_head  = ['R','C']
        additional_ins  = ['resistor','capacitor']
        additional_ins_val  = ['r','c']
        additional_ins_unit  = ['*1K','*1p']
        additional_ins_bound  = ['80','1']
    
    return instance_head, node, instance, instance_val, unit, bound, additional_node, additional_ins_head, additional_ins, additional_ins_val, additional_ins_unit, additional_ins_bound

def amp_generator(design_id, topo_vector):
    
    os.chdir('tmp_circuit')
    os.system('mkdir '+design_id)
    os.chdir(design_id)
    os.system('mkdir circuit')
    os.chdir('circuit')
    os.system('mkdir netlist')
    os.chdir('../../..')
        
    netlistFooter = open('./tmp_circuit/'+design_id+'/circuit/netlist/netlistFooter', 'w')
    netlistFooter.writelines('// Footer end\n')
    netlistFooter.close()
    netlistHeader = open('./tmp_circuit/'+design_id+'/circuit/netlist/netlistHeader', 'w')
    netlistHeader.writelines('// Generated by amp topo opt program\n')
    netlistHeader.writelines('simulator lang=spectre\n')
    netlistHeader.writelines('global 0\n')
    netlistHeader.writelines('// Header end\n')
    netlistHeader.close()
    
    tmp_conf = open('./tmp_circuit/'+design_id+'/tmp.conf', 'w')
    netlist  = open('./tmp_circuit/'+design_id+'/circuit/netlist/netlist', 'w')
    
    netlist.writelines('V0 (net_vin 0) vsource mag=1 type=sine\n')     # V_source setting

    netlist.writelines('G_vin_1 (net_1 0 net_vin 0) vccs gm=g_vin_1*1m\n')
    tmp_conf.writelines('des_var g_vin_1 0.01 1\n')
    netlist.writelines('R_1_0_prs (net_1 0) resistor r=rscl_vin_1_prs/g_vin_1*1K\n')
    tmp_conf.writelines('des_var rscl_vin_1_prs 40 80\n')
    netlist.writelines('C_1_0_prs (net_1 0) capacitor c=g_1_2/6.28*5p\n')

    netlist.writelines('G_1_2 (net_2 0 net_1 0) vccs gm=g_1_2*-1m\n')
    tmp_conf.writelines('des_var g_1_2 0.01 1\n')
    netlist.writelines('R_2_0_prs (net_2 0) resistor r=rscl_1_2_prs/g_1_2*1K\n')
    tmp_conf.writelines('des_var rscl_1_2_prs 40 80\n')
    netlist.writelines('C_2_0_prs (net_2 0) capacitor c=g_2_vo/6.28*5p\n')

    netlist.writelines('G_2_vo (net_vo 0 net_2 0) vccs gm=g_2_vo*1m\n')
    tmp_conf.writelines('des_var g_2_vo 0.01 1\n')
    netlist.writelines('R_vo_0_prs (net_vo 0) resistor r=rscl_2_vo_prs/g_2_vo*1K\n')
    tmp_conf.writelines('des_var rscl_2_vo_prs 40 80\n')
    #netlist.writelines('C_vo_0_prs (net_vo 0) capacitor c=c_vo_0_prs*1p\n')
    #tmp_conf.writelines('des_var c_vo_0_prs 0.01 10\n')

    netlist.writelines('R_L (net_vo 0) resistor r=0.15M\n')         # RC load setting
    netlist.writelines('C_L (net_vo 0) capacitor c=10n\n')

    node_pairs = [['_vin','_2','_vin_2'],['_vin','_vo','_vin_vo'],['_1','_vo','_1_vo'],['_1','_0','_1_0'],['_2','_0','_2_0']]

    for idx,node in enumerate(node_pairs):
        x = topo_vector[idx,0]
        node_pair = node_pairs[idx]
        instance_head, node, instance, instance_val, unit, bound, additional_node, additional_ins_head, additional_ins, additional_ins_val, additional_ins_unit, additional_ins_bound = get_node_info(x)
        if x==0:
            pass
        elif 1<=x<=2:
            if idx <=2:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')                
        elif x==3:
            if idx <=2:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')                
        elif x==4:
            if idx <=2:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' (net'+node_pair[node[0]]+' net'+node_pair[additional_node[0]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' (net'+node_pair[node[0]]+' net'+node_pair[additional_node[0]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')                
        elif x in [15,16]:
            netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
            tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')

            netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
            netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
        elif x in [5,6]: 
            netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
            tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')

            netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
            netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
            # add input parasitic capacitor
            netlist.writelines('C_input_'+str(int(idx))+' (net'+node_pair[node[1]]+' 0) capacitor c='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+'/6.28*5p'+'\n')
        elif 7<=x<=14:
            if 11<=x<=14:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
                # add input parasitic capacitor
                netlist.writelines('C_input_'+str(int(idx))+' (net'+node_pair[node[1]]+' 0) capacitor c='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+'/6.28*5p'+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[additional_node[0]]+node_pair[node[0]]+' (net'+node_pair[additional_node[0]]+' 0 net'+node_pair[node[0]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
                # add input parasitic capacitor
                netlist.writelines('C_input_'+str(int(idx))+' (net'+node_pair[node[0]]+' 0) capacitor c='+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+'/6.28*5p'+'\n')
        elif 17<=x<=24:
            if x%2!=0:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' (net'+node_pair[node[0]]+' net'+node_pair[additional_node[0]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
        else:
            print('error topo input !')

    netlist.close()
    tmp_conf.close()

def dac21_amp_generator(design_id, topo_vector):
    
    os.chdir('dac21_tmp_circuit')
    os.system('mkdir '+design_id)
    os.chdir(design_id)
    os.system('mkdir circuit')
    os.chdir('circuit')
    os.system('mkdir netlist')
    os.chdir('../../..')
        
    netlistFooter = open('./dac21_tmp_circuit/'+design_id+'/circuit/netlist/netlistFooter', 'w')
    netlistFooter.writelines('// Footer end\n')
    netlistFooter.close()
    netlistHeader = open('./dac21_tmp_circuit/'+design_id+'/circuit/netlist/netlistHeader', 'w')
    netlistHeader.writelines('// Generated by amp topo opt program\n')
    netlistHeader.writelines('simulator lang=spectre\n')
    netlistHeader.writelines('global 0\n')
    netlistHeader.writelines('// Header end\n')
    netlistHeader.close()
    
    tmp_conf = open('./dac21_tmp_circuit/'+design_id+'/tmp.conf', 'w')
    netlist  = open('./dac21_tmp_circuit/'+design_id+'/circuit/netlist/netlist', 'w')
    
    netlist.writelines('V0 (net_vin 0) vsource mag=1 type=sine\n')     # V_source setting

    netlist.writelines('G_vin_1 (net_1 0 net_vin 0) vccs gm=g_vin_1*1m\n')
    tmp_conf.writelines('des_var g_vin_1 0.01 1\n')
    netlist.writelines('R_1_0_prs (net_1 0) resistor r=rscl_vin_1_prs/g_vin_1*1K\n')
    tmp_conf.writelines('des_var rscl_vin_1_prs 40 80\n')
    netlist.writelines('C_1_0_prs (net_1 0) capacitor c=g_1_2/6.28*5p\n')

    netlist.writelines('G_1_2 (net_2 0 net_1 0) vccs gm=g_1_2*-1m\n')
    tmp_conf.writelines('des_var g_1_2 0.01 1\n')
    netlist.writelines('R_2_0_prs (net_2 0) resistor r=rscl_1_2_prs/g_1_2*1K\n')
    tmp_conf.writelines('des_var rscl_1_2_prs 40 80\n')
    netlist.writelines('C_2_0_prs (net_2 0) capacitor c=g_2_vo/6.28*5p\n')

    netlist.writelines('G_2_vo (net_vo 0 net_2 0) vccs gm=g_2_vo*1m\n')
    tmp_conf.writelines('des_var g_2_vo 0.01 1\n')
    netlist.writelines('R_vo_0_prs (net_vo 0) resistor r=rscl_2_vo_prs/g_2_vo*1K\n')
    tmp_conf.writelines('des_var rscl_2_vo_prs 40 80\n')
    #netlist.writelines('C_vo_0_prs (net_vo 0) capacitor c=c_vo_0_prs*1p\n')
    #tmp_conf.writelines('des_var c_vo_0_prs 0.01 10\n')

    netlist.writelines('R_L (net_vo 0) resistor r=0.15M\n')         # RC load setting
    netlist.writelines('C_L (net_vo 0) capacitor c=10n\n')

    node_pairs = [['_vin','_2','_vin_2'],['_vin','_vo','_vin_vo'],['_1','_vo','_1_vo'],['_1','_0','_1_0'],['_2','_0','_2_0']]

    for idx,node in enumerate(node_pairs):
        x = topo_vector[idx,0]
        node_pair = node_pairs[idx]
        instance_head, node, instance, instance_val, unit, bound, additional_node, additional_ins_head, additional_ins, additional_ins_val, additional_ins_unit, additional_ins_bound = get_node_info(x)
        if x==0:
            pass
        elif 1<=x<=2:
            if idx <=2:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')                
        elif x==3:
            if idx <=2:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')                
        elif x==4:
            if idx <=2:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' (net'+node_pair[node[0]]+' net'+node_pair[additional_node[0]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' (net'+node_pair[node[0]]+' net'+node_pair[additional_node[0]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.5 '+bound[1]+'\n')                
        elif x in [15,16]:
            netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
            tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')

            netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
            netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
        elif x in [5,6]: 
            netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
            tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')

            netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
            netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
            tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
            # add input parasitic capacitor
            netlist.writelines('C_input_'+str(int(idx))+' (net'+node_pair[node[1]]+' 0) capacitor c='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+'/6.28*5p'+'\n')
        elif 7<=x<=14:
            if 11<=x<=14:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
                # add input parasitic capacitor
                netlist.writelines('C_input_'+str(int(idx))+' (net'+node_pair[node[1]]+' 0) capacitor c='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+'/6.28*5p'+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[additional_node[0]]+node_pair[node[0]]+' (net'+node_pair[additional_node[0]]+' 0 net'+node_pair[node[0]]+' 0) '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' net'+node_pair[node[1]]+') '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
                # add input parasitic capacitor
                netlist.writelines('C_input_'+str(int(idx))+' (net'+node_pair[node[0]]+' 0) capacitor c='+instance_val[0]+node_pair[additional_node[0]]+node_pair[node[0]]+'/6.28*5p'+'\n')
        elif 17<=x<=24:
            if x%2!=0:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' net'+node_pair[node[1]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[node[0]]+node_pair[node[1]]+' (net'+node_pair[node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[1]+node_pair[node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
            else:
                netlist.writelines(instance_head[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' (net'+node_pair[node[0]]+' net'+node_pair[additional_node[0]]+') '+instance[0]+' '+instance_val[0]+'='+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+unit[0]+'\n')
                tmp_conf.writelines('des_var '+instance_val[0]+node_pair[node[0]]+node_pair[additional_node[0]]+' 0.01 '+bound[0]+'\n')
                netlist.writelines(instance_head[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' (net'+node_pair[additional_node[0]]+' 0 net'+node_pair[node[1]]+' 0) '+instance[1]+' '+instance_val[1]+'='+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+unit[1]+'\n')
                tmp_conf.writelines('des_var '+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+' 0.01 '+bound[1]+'\n')
                
                netlist.writelines(additional_ins_head[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[0]+' '+additional_ins_val[0]+'='+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+'/'+instance_val[1]+node_pair[additional_node[0]]+node_pair[node[1]]+additional_ins_unit[0]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[0]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 40 '+additional_ins_bound[0]+'\n')
                netlist.writelines(additional_ins_head[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' (net'+node_pair[additional_node[0]]+' 0) '+additional_ins[1]+' '+additional_ins_val[1]+'='+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+additional_ins_unit[1]+'\n')
                tmp_conf.writelines('des_var '+additional_ins_val[1]+node_pair[additional_node[0]]+'_0'+str(int(idx))+' 0.5 '+additional_ins_bound[1]+'\n')
        else:
            print('error topo input !')

    netlist.close()
    tmp_conf.close()
