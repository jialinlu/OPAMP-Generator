# optimization usage functions

import os
import re
import math
import random
import string
import numpy as np
from netlist_generator import *
from util import *
#from models import *
from dataset import *

def gen_random_id():
    random_id = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    return random_id

def optimize_circuit(design_id,scratch):
    work_dir = os.getcwd()
    os.chdir('./tmp_circuit/'+design_id+'/circuit/')
    os.system('cp '+work_dir+'/template/run.pl ./')
    os.system('cp '+work_dir+'/template/clear.sh ./')
    os.system('chmod u+x *')
    os.system('cp '+work_dir+'/template/extract.py ./')
    os.system('cp '+work_dir+'/template/ocnScript_generate.py ./')
    os.system('cp '+work_dir+'/template/ocnScript_temp.ocn ./')
    #os.system('cp -r '+work_dir+'/template/model ./')
    os.chdir('..')
    pwd = os.getcwd()
    conf = open('./conf','w')
    conf.writelines('workdir '+pwd+'\n')
    conf.close()
    if scratch == True:
        os.system('cat ./tmp.conf '+work_dir+'/template/temp_scratch.conf >> ./conf')
    else:
        os.system('cat ./tmp.conf '+work_dir+'/template/temp.conf >> ./conf')
    os.system('weibo ./conf > log 2>err')
    os.chdir('../..')

def evaluate_topo(topo_vector, cons_num, scratch):
    topo_dims   = len(topo_vector)
    topo_vector = topo_vector.reshape(topo_dims,1)
    design_id   = gen_random_id()
    amp_generator(design_id, topo_vector)
    optimize_circuit(design_id, scratch)

    res   = open('./tmp_circuit/'+design_id+'/circuit/result.po', 'r')
    lines = res.readlines()
    cut   = lines[0].split()
    goal = float(cut[0])
    cons = np.ones((cons_num,1))
    for i in range(cons_num):
        cons[i,0] = float(cut[i+1])
    res.close()

    return goal, cons, design_id

def topo2emb(topo_vector, model, device):
    design_id = gen_random_id()
    row = topo2graph(topo_vector)
    g0, _ = decode_CIRCUIT_to_igraph(row)
    emb, _ = model.encode(g0)
    return emb.cpu().detach().numpy(), design_id

def topo2graph(topo_vector):

    
    CIRCUIT_DAG = []
    for i in range(CIRCUIT_NODE_NUM):
        if i == 0:
            CIRCUIT_DAG.append([0]) # node Vin
        elif i == 1:
            CIRCUIT_DAG.append([1,5]) # node 1
        elif i == 2:
            CIRCUIT_DAG.append([1,topo_vector[0,0],6]) # node 2
        elif i == 3:
            CIRCUIT_DAG.append([2,topo_vector[1,0],topo_vector[2,0],5]) # node Vout
        elif i == 4:
            CIRCUIT_DAG.append([3,0,topo_vector[3,0],topo_vector[4,0],0]) # node GND
        else:
            pass
    return CIRCUIT_DAG

def pdf(x):
    # x = (x-mu)/theta
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)

def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911                      
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)                                     
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)                                                   
    return sign*y

def cdf(x):
    # x = (x-mu)/theta
    return 0.5+erf(x/np.sqrt(2))/2

def logphi(x):
    if x**2 < 0.0492:
        lp0 = -x/np.sqrt(2*np.pi)
        c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138, 0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021, 2*(1-np.pi/3), (4-np.pi)/3, 1, 1])
        f = 0
        for i in range(14):
            f = lp0*(c[i]+f)
        return -2*f-np.log(2)
    elif x < -11.3137:
        r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425, 2.9788656263939928886])
        q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677])
        num = 0.5641895835477550741
        for i in range(5):
            num = -x*num/np.sqrt(2)+r[i]
        den = 1.0
        for i in range(6):
            den = -x*den/np.sqrt(2)+q[i]
        return np.log(0.5*np.maximum(0.000001,num/den))-0.5*(x**2)
    else:
        return np.log(0.5*np.maximum(0.000001,(1.0-erf(-x/np.sqrt(2)))))
