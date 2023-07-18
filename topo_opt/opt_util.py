# optimization usage functions

import os
import re
import math
import random
import string
import numpy as np
import pandas as pd
from netlist_generator import *
from util import *
# from models import *
from dataset import *
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO


def gen_random_id():
    random_id = ''.join(random.sample(string.ascii_letters + string.digits, 6))
    return random_id


def optimize_circuit(design_id, scratch):
    work_dir = os.getcwd()
    os.chdir('./tmp_circuit/' + design_id + '/circuit/')
    # os.system('cp ' + work_dir + '/template/run.pl ./')
    os.system('cp ' + work_dir + '/template/clear.sh ./')
    os.system('chmod u+x *')
    os.system('cp ' + work_dir + '/template/extract.py ./')
    os.system('cp ' + work_dir + '/template/ocnScript_generate.py ./')
    os.system('cp ' + work_dir + '/template/ocnScript_temp.ocn ./')
    # os.system('cp -r '+work_dir+'/template/model ./')
    os.chdir('..')
    pwd = os.getcwd()
    conf = open('./conf', 'w')
    conf.writelines('workdir ' + pwd + '\n')
    conf.close()
    if scratch == True:
        os.system('cat ./tmp.conf ' + work_dir + '/template/temp_scratch.conf >> ./conf')
    else:
        os.system('cat ./tmp.conf ' + work_dir + '/template/temp.conf >> ./conf')
    name = []
    bounds = np.zeros((2, 0))
    with open('./conf', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 0 and line[0] == 'des_var':
                name.append(line[1])
                bounds = np.hstack((bounds, np.array([float(line[2]), float(line[3])]).reshape(-1, 1)))
    # os.system('weibo ./conf > log 2>err')
    problem = Opamp(design_id, name, bounds)
    _, _, best_y, _ = opt(problem)
    os.chdir('../..')
    return best_y


class Opamp():
    def __init__(self,
                 design_id: string,
                 name: list,
                 bounds: np.ndarray,
                 num_obj: int = 1,
                 num_constr: int = 4
                 ):
        self.design_id = design_id
        self.name = name
        self.n_var = bounds.shape[1]
        self.n_obj = num_obj
        self.n_constr = num_constr
        self.bounds = bounds

    def _evaluate(self, x: np.ndarray, *args, **kwargs):
        num_x = x.shape[0]
        os.chdir('./circuit')
        ocn_script_file = './oceanScript_opamp.ocn'
        result_file = './result.po'

        out = np.zeros((num_x, self.n_obj + self.n_constr))

        temp = open('./ocnScript_temp.ocn', 'r')
        temp_lines = temp.readlines()

        for i in range(num_x):
            os.system('./clear.sh')
            param = open('./param', 'w')
            with open(ocn_script_file, 'w') as f:
                for k in range(7):
                    f.writelines(temp_lines[k])
                for j in range(self.n_var):
                    f.writelines('desVar( ' + '"' + self.name[j] + '" ' + str(x[i][j]) + ' )\n')
                    param.writelines('.param ' + self.name[j] + ' = ' + str(x[i][j]) + '\n')

                for k in range(7, 31):
                    f.writelines(temp_lines[k])

            param.close()

            os.system('ocean -replay ./oceanScript_opamp.ocn -log ocean.log > err 2>&1')
            # os.system('ocean -replay oceanScript_opamp.ocn')
            os.system('python extract.py')
            os.system('cat param result.po >> backup')

            with open(result_file, 'r') as f:
                perf = f.readlines()[0].strip().split()
            out[i] = np.array([float(p) for p in perf])
        temp.close()
        os.chdir('..')
        return out


def opt(problem, rand_sample=10, iter=40, batch=1):
    dim = problem.n_var
    num_obj = problem.n_obj
    num_constr = problem.n_constr

    def obj(param: pd.DataFrame) -> (np.ndarray, np.ndarray):
        names = ['x' + str(i) for i in range(problem.n_var)]
        x = param[names].values
        out = problem._evaluate(x)
        return out

    lb, ub = problem.bounds
    params = [{'name': 'x' + str(i), 'type': 'num', 'lb': lb[i], 'ub': ub[i]} for i in range(dim)]
    space = DesignSpace().parse(params)

    conf = {}
    conf['num_epochs'] = 100
    opt = GeneralBO(space, num_obj, num_constr, model_conf=conf, rand_sample=rand_sample)
    for i in range(iter):
        rec = opt.suggest(n_suggestions=batch)
        perf = obj(rec)
        opt.observe(rec, perf)

    if num_constr > 0:
        cons = np.sum(np.maximum(np.array(opt.y)[:, 1:], 0), axis=1)
        # print(cons)
        if np.min(cons) == 0:
            cand = np.where(cons == 0)
            feasible_x = np.array(opt.X)[cand]
            feasible_y = np.array(opt.y)[cand]
            # print(feasible_y)
            best_id = np.argmin(feasible_y[:, 0])
            best_y, best_x = feasible_y[best_id], feasible_x[best_id]
        else:
            best_id = np.argmin(cons)
            best_y, best_x = np.array(opt.y)[best_id], np.array(opt.X)[best_id]
    else:
        best_id = np.argmin(np.array(opt.y).flatten())
        best_y, best_x = np.array(opt.y)[best_id], np.array(opt.X)[best_id]
    return np.array(opt.y), np.array(opt.X), best_y, best_x


def evaluate_topo(topo_vector, cons_num, scratch):
    topo_dims = len(topo_vector)
    topo_vector = topo_vector.reshape(topo_dims, 1)
    design_id = gen_random_id()
    amp_generator(design_id, topo_vector)
    best_y = optimize_circuit(design_id, scratch)
    goal = best_y[0]
    cons = best_y[1:].reshape(-1, 1)
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
            CIRCUIT_DAG.append([0])  # node Vin
        elif i == 1:
            CIRCUIT_DAG.append([1, 5])  # node 1
        elif i == 2:
            CIRCUIT_DAG.append([1, topo_vector[0, 0], 6])  # node 2
        elif i == 3:
            CIRCUIT_DAG.append([2, topo_vector[1, 0], topo_vector[2, 0], 5])  # node Vout
        elif i == 4:
            CIRCUIT_DAG.append([3, 0, topo_vector[3, 0], topo_vector[4, 0], 0])  # node GND
        else:
            pass
    return CIRCUIT_DAG


def pdf(x):
    # x = (x-mu)/theta
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


def erf(x):
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x ** 2)
    return sign * y


def cdf(x):
    # x = (x-mu)/theta
    return 0.5 + erf(x / np.sqrt(2)) / 2


def logphi(x):
    if x ** 2 < 0.0492:
        lp0 = -x / np.sqrt(2 * np.pi)
        c = np.array([0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802, 0.00556964649138,
                      0.00125993961762116, -0.01621575378835404, 0.02629651521057465, -0.001829764677455021,
                      2 * (1 - np.pi / 3), (4 - np.pi) / 3, 1, 1])
        f = 0
        for i in range(14):
            f = lp0 * (c[i] + f)
        return -2 * f - np.log(2)
    elif x < -11.3137:
        r = np.array([1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441, 7.409740605964741794425,
                      2.9788656263939928886])
        q = np.array([2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034, 17.081440747466004316,
                      9.608965327192787870698, 3.3690752069827527677])
        num = 0.5641895835477550741
        for i in range(5):
            num = -x * num / np.sqrt(2) + r[i]
        den = 1.0
        for i in range(6):
            den = -x * den / np.sqrt(2) + q[i]
        return np.log(0.5 * np.maximum(0.000001, num / den)) - 0.5 * (x ** 2)
    else:
        return np.log(0.5 * np.maximum(0.000001, (1.0 - erf(-x / np.sqrt(2)))))
