# amplifier topology optimization with gae and weibo
# weibo algorithm can refer to “An Efficient Bayesian Optimization Approach for Automated Optimization of Analog Circuits”
# the acquisition function solving method was switched from ga to bfgs

import os
import sys
import argparse
import logging
import time
import math
import traceback
import GPy
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from shutil import copy
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

import igraph
import torch
from torch import nn, optim

from util import *
from dataset import *
from opt_util import *

parser = argparse.ArgumentParser("Topology Optimization for Amplifier with GAE")
# general settings
parser.add_argument('--data_name', default='threeStageOpamp', help='optimization results save name')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--backup', action='store_true', default=True,
                    help='if True, copy current py files to result dir')
# model settings
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
parser.add_argument('--model', default='DVAE_test2', help='model to use')
parser.add_argument('--hs', type=int, default=501, metavar='N', help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=10, metavar='N', help='number of dimensions of latent vectors z')
parser.add_argument('--load_model_path', default='_nz10', help='model path to loaded')
parser.add_argument('--load_model_name', default='200', help='model name to loaded')
parser.add_argument('--which_gp', type=str, default='sklearn', help='use which gp to build, GPy or sklearn')
# optimization settings
parser.add_argument('--iteration', type=int, default=2, help='total iteratons nums')
parser.add_argument('--init_num',  type=int, default=2,  help='random nums of samplings for optimization')
parser.add_argument('--random_sampling', action='store_true', default=False,
                    help='whether do random sampling during upper level optimization')
parser.add_argument('--emb_bound',  type=int, default=50,  help='the bound setting of emb value for fmin_l_bfgs_b')
parser.add_argument('--bfgs_time',  type=int, default=10,  help='the exec times of bfgs solving')
parser.add_argument('--bfgs_debug',  type=bool, default=False,  help='the debug information of bfgs')
parser.add_argument('--samping_ratio',  type=float, default=0.01,  help='the ratio of samplings around current best point')

# circuit settings
parser.add_argument('--topo_dims', type=int, default=5, help='vector dims of amp topo params, which is the CIRCUIT_NODE_NUM')
parser.add_argument('--constr_num', type=int, default=4, help='constraint nums of circuits')

args = parser.parse_args()

# save files and logs
args.file_dir = os.getcwd()
args.res_dir  = os.path.join(args.file_dir, 'opt_results/{}{}'.format(args.data_name, args.save_appendix))

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.res_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(args)

cmd_input = 'python ' + ' '.join(sys.argv)
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
logging.info('command line input: ' + cmd_input + ' is saved.')

if args.backup:
    copy('optimization_bfgs.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('util.py', args.res_dir)
    copy('opt_util.py', args.res_dir)

# load trained autoencoder model
gpu = 'cuda:'+str(args.gpu)
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

load_model_path  = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, args.load_model_path))
load_model_name = 'model_checkpoint' + args.load_model_name + '.pth'
model = torch.load(os.path.join(load_model_path, load_model_name))
model.to(device)
logging.info('model: {} has been loaded'.format(os.path.join(load_model_path, load_model_name)))

# load init data
iteration  = args.iteration
constr_num = args.constr_num
init_num   = args.init_num
topo_dims  = args.topo_dims
solutions  = {}

logging.info('collecting init data ...')
sul = open(os.path.join(args.res_dir, 'solutions.txt'),'w')

if args.random_sampling:
    logging.info('begin init...')
    initX_topo   = np.zeros((init_num, topo_dims))
    initY        = np.zeros((init_num, 1))
    initConstr   = np.zeros((init_num, constr_num))
    for i in range(init_num):
        random_topo = np.zeros((topo_dims,1))
        for j in range(topo_dims):
            if j<=1:
                random_topo[j,0] = np.random.randint(0,11)
            elif j==2:
                random_topo[j,0] = np.random.randint(0,25)
            else:
                random_topo[j,0] = np.random.randint(0,5)
        goal, constr, design_id = evaluate_topo(random_topo, constr_num, False)
        
        initX_topo[i] = random_topo.reshape(topo_dims)
        initY[i,0]    = goal
        initConstr[i] = constr.reshape(constr_num)
        logging.info('design_id: '+design_id)
        logging.info('topo: '+str(random_topo[:,0]))
        logging.info('goal: '+str(goal))
        logging.info('constr: '+str(constr[:,0]))
        logging.info(str(i)+'-th init finished')

        if constr[0,0]<=0 and constr[1,0]<=0 and constr[2,0]<=0 and constr[3,0]<=0 :
            solutions.update({design_id: random_topo[:,0]})
            sul.writelines(design_id+': '+str(random_topo[:,0])+'\n')
            sul.writelines('goal: '+str(goal)+'\n')
            sul.writelines('constr: '+str(constr[:,0])+'\n')
            sul.writelines('###\n')

    np.savetxt('./init_data/initX_topo.csv', initX_topo, fmt='%d', delimiter=',')
    np.savetxt('./init_data/initConstr.csv', initConstr, fmt='%d', delimiter=',')
    np.savetxt('./init_data/initY.csv', initY, fmt='%f', delimiter=',')
    logging.info('all initialization finished ...')

else:
    logging.info('loading existing init data...')
    initX_topo = np.loadtxt('./init_data/initX_topo.csv', delimiter=',').astype(np.int8)
    initConstr = np.loadtxt('./init_data/initConstr.csv', delimiter=',').astype(np.float32)
    initY      = np.loadtxt('./init_data/initY.csv', delimiter=',').astype(np.float32)
    logging.info('init dataset size: {}'.format(len(initY)))

Xs_topo = np.reshape(initX_topo, (-1,topo_dims))
Ys      = np.reshape(initY, (-1,1))
Constr0 = np.reshape(initConstr[:,0],(-1, 1))
Constr1 = np.reshape(initConstr[:,1],(-1, 1))
Constr2 = np.reshape(initConstr[:,2],(-1, 1))
Constr3 = np.reshape(initConstr[:,3],(-1, 1))

'''some util functions for optimization'''
if args.which_gp == 'sklearn': # using sklearn
    logging.info('using sklearn lib to build GP models...')
    def gp_modeling (x,y):
        #lengthScale = np.std(x,0)
        kernel = C(1.0, (1e-3, 1e3)) * RBF()
        gp_model = GaussianProcessRegressor(alpha=0.1, kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gp_model.fit(x,y)
        return gp_model
else: # using GPy
    logging.info('using GPy lib to build GP models...')
    def gp_modeling (x,y):
        k = GPy.kern.RBF(args.nz)
        gp_model = GPy.models.GPRegression(x,y)
        gp_model.optimize('bfgs')
        return gp_model

def wEI(x, models, best_y):
    x = np.array(x).reshape(1,args.nz)
    for i in range(len(models)):
        if args.which_gp=='sklearn':
            py, ps2 = models[i].predict(x, return_std=True)
        else:
            py, ps2 = models[i].predict(x)
        py = py.sum()
        ps = np.sqrt(np.diag(ps2))
        if i == 0: # EI
            tmp = -(py - best_y) / ps
            tmp_loss = ps * (tmp * cdf(tmp) + pdf(tmp))
        else: # PI
            tmp_loss = tmp_loss * cdf(-py/ps)
    return tmp_loss

def log_wEI(x, models, best_y):
    x = np.array(x).reshape(1,args.nz)
    EI = 1.0
    if args.which_gp=='sklearn':
        py, ps2 = models[0].predict(x, return_std=True)
    else:
        py, ps2 = models[0].predict(x)
    py = py.sum()
    ps = np.sqrt(ps2.sum())
    tmp = (best_y - py)/ps
    EI = ps*(tmp*cdf(tmp)+pdf(tmp))
    EI = np.log(np.maximum(0.000001, EI))
    PI = 1.0
    for i in range(1, len(models)):
        if args.which_gp=='sklearn':
            py, ps2 = models[i].predict(x, return_std=True)
        else:
            py, ps2 = models[i].predict(x)
        py = py.sum()
        ps = np.sqrt(ps2.sum())
        # PI = PI*cdf(-py/ps)
        PI = PI + logphi(-py/ps)
    return EI + PI

def bfgs(models, wEI_func, best_y, new_x):
    bounds = []
    for i in range(args.nz):
        bounds.append([-1*args.emb_bound,args.emb_bound])
    bounds = np.array(bounds)
    sampling_prob = np.random.uniform(0,1)
    if sampling_prob>args.samping_ratio: # sampling around the current best
        x0 = np.random.uniform(-1*args.emb_bound, args.emb_bound, (args.nz))
    else:
        x0 = np.random.uniform(-1, 1, (args.nz)) + new_x.reshape((args.nz))
    best_x = np.copy(x0)
    best_loss = np.inf

    def loss(x):
        nonlocal best_x
        nonlocal best_loss
        wEI_loss = wEI_func(x, models, best_y)
        #wEI_loss = -wEI_loss.sum()
        wEI_loss = -wEI_loss
        if wEI_loss < best_loss:
            best_loss = wEI_loss
            best_x = np.copy(x)
        return wEI_loss

    fmin_l_bfgs_b(loss, x0, bounds=bounds, approx_grad=True, maxiter=1000, m=100, iprint=args.bfgs_debug)
    if(np.isnan(best_loss) or np.isinf(best_loss)):
        logging.info('Fail to build GP model')
        sys.exit(1)
    return best_x

# optomization begin here

best_y       = 0
best_topo    = []
best_constrs = []
new_x        = np.random.uniform(-1*args.emb_bound, args.emb_bound, (args.nz))

flow_begin = time.time()

for cnt in range(iteration):

    logging.info('-'*20+'ITERATION '+str(cnt)+'-'*20)

    y1 = Ys.reshape(len(Ys),1)
    x1 = np.zeros((len(Xs_topo), args.nz))
    c0 = Constr0.reshape(len(Constr0),1)
    c1 = Constr1.reshape(len(Constr1),1)
    c2 = Constr2.reshape(len(Constr2),1)
    c3 = Constr3.reshape(len(Constr3),1)

    for i in range(len(Xs_topo)):
        topo  = Xs_topo[i].reshape(topo_dims,1)
        emb,design_id   = topo2emb(topo, model, device)
        x1[i] = np.reshape(emb,args.nz)

    gp_models  = []
    gp_goal    = gp_modeling(x1,y1)
    gp_models.append(gp_goal)
    gp_constr0 = gp_modeling(x1,c0)
    gp_models.append(gp_constr0)
    gp_constr1 = gp_modeling(x1,c1)
    gp_models.append(gp_constr1)
    gp_constr2 = gp_modeling(x1,c2)
    gp_models.append(gp_constr2)
    gp_constr3 = gp_modeling(x1,c3)
    gp_models.append(gp_constr3)

    best_wEI = -np.inf
    logging.info('executing fmin_l_bfgs_b...')
    opt_begin = time.time()
    for i in range(args.bfgs_time):
        best_x = bfgs(gp_models, log_wEI, best_y, new_x) # use log(wEI)
        tmp_wEI = wEI(best_x, gp_models, best_y)
        if tmp_wEI > best_wEI:
            best_wEI = tmp_wEI
            new_x = best_x
    opt_end = time.time()
    logging.info('wEI function optimization finished, cost {} seconds'.format(opt_end-opt_begin))

    new_x = new_x.reshape(args.nz,1)
    if args.which_gp=='sklearn':
        py_goal, ps2_goal = gp_goal.predict(new_x.reshape(1, args.nz), return_std=True)
        py_constr0, ps2_constr0 = gp_constr0.predict(new_x.reshape(1, args.nz), return_std=True)
        py_constr1, ps2_constr1 = gp_constr1.predict(new_x.reshape(1, args.nz), return_std=True)
        py_constr2, ps2_constr2 = gp_constr2.predict(new_x.reshape(1, args.nz), return_std=True)
        py_constr3, ps2_constr3 = gp_constr3.predict(new_x.reshape(1, args.nz), return_std=True)
    else:
        py_goal, ps2_goal = gp_goal.predict(new_x.reshape(1, args.nz))
        py_constr0, ps2_constr0 = gp_constr0.predict(new_x.reshape(1, args.nz))
        py_constr1, ps2_constr1 = gp_constr1.predict(new_x.reshape(1, args.nz))
        py_constr2, ps2_constr2 = gp_constr2.predict(new_x.reshape(1, args.nz))
        py_constr3, ps2_constr3 = gp_constr3.predict(new_x.reshape(1, args.nz))

    new_g = model.decode(torch.tensor(new_x).reshape(1,args.nz).to(device).float())
    new_topo = g2topo(new_g[0])
    logging.info('new_Topo: {}'.format(new_topo))
    logging.info('gp_goal_prediction: {}'.format(py_goal[0][0]))
    logging.info('gp_constr_prediction: {}, {}, {}, {}'.format(py_constr0[0][0],py_constr1[0][0],py_constr2[0][0],py_constr3[0][0]))
    logging.info('wEI_value: {}'.format(best_wEI[0][0]))
    #time.sleep(100)

    logging.info('sizing begin ...')
    sizing_begin = time.time()
    new_goal, new_constr, new_design_id = evaluate_topo(new_topo, constr_num, False)
    sizing_end = time.time()
    logging.info('sizing finished, cost {} minutes'.format((sizing_end-sizing_begin)/60))
    logging.info('new_design_id: {}'.format(new_design_id))
    logging.info('new_goal: {}'.format(new_goal))
    logging.info('new_constr: {}'.format(new_constr[:,0]))

    if new_constr[0,0]<=0 and new_constr[1,0]<=0 and new_constr[2,0]<=0 and new_constr[3,0]<=0 :
        solutions.update({new_design_id: new_topo})
        sul.writelines(new_design_id+': '+str(new_topo)+'\n')
        sul.writelines('goal: {}'.format(new_goal)+'\n')
        sul.writelines('constr: {}'.format(new_constr[:,0])+'\n')
        sul.writelines('###\n')

        logging.info('yes!!! find one satisfied circuit in round {}'.format(cnt))
        logging.info('already found {} circuits satisfied all constrains'.format(len(solutions)))

        if new_goal < best_y:
            best_y = new_goal
            best_topo = list(new_topo)
            best_constrs = [new_constr[0,0], new_constr[1,0], new_constr[2,0], new_constr[3,0]]

    logging.info('###')
    logging.info('current best y: {}'.format(best_y))
    logging.info('corresponding topo is: {}'.format(best_topo))
    logging.info('corresponding constrains are: {}'.format(best_constrs))

    # update the dataset
    Ys      = np.vstack((Ys, new_goal))
    Xs_topo = np.vstack((Xs_topo, new_topo.reshape(topo_dims)))
    Constr0 = np.vstack((Constr0,new_constr[0,0]))
    Constr1 = np.vstack((Constr1,new_constr[1,0]))
    Constr2 = np.vstack((Constr2,new_constr[2,0]))
    Constr3 = np.vstack((Constr3,new_constr[3,0]))

    Xs_topo = np.reshape(Xs_topo, (-1,topo_dims))

np.savetxt(os.path.join(args.res_dir, 'Xs_topo_res.csv'), Xs_topo, fmt='%d', delimiter=',')
np.savetxt(os.path.join(args.res_dir, 'Ys_res.csv'), Ys, fmt='%f', delimiter=',')
sul.close()

flow_end = time.time()

logging.info('full optimization finished, cost {} hours'.format((flow_end-flow_begin)/3600))
logging.info('found {} circuits satisfied all constrains totally !'.format(len(solutions)))
