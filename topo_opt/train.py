import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models import *
from dataset import *

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-name', default='threeStageOpamp', help='graph dataset name')
parser.add_argument('--save-appendix', default='', 
                    help='what to append to data-name as save-name for results')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--backup', action='store_true', default=True,
                    help='if True, copy current py files to result dir')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=10, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
# training settings
parser.add_argument('--model', default='DVAE_test2', help='model to use')
parser.add_argument('--data_file', type=str, default='dataset_withoutY', help='dataset original file to use')
parser.add_argument('--trainSet_size', type=int, default=10000, help='control the size of training set')
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=10, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--load_model_path', default='_nz10_10w', help='model path to loaded')
parser.add_argument('--load_model_name', default='200', help='model name to loaded')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)
gpu = 'cuda:'+str(args.gpu)
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare data'''
args.file_dir = os.getcwd()
args.res_dir  = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, 
                                                                 args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name):
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
# otherwise process the raw data and save to .pkl
else:
    data_file = args.data_file
    train_data, test_data, graph_args = load_CIRCUIT_graphs(data_file)
    train_data = train_data[:args.trainSet_size]
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)

if args.backup:
    # backup current .py files
    copy('train.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('util.py', args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

'''Prepare the model'''
# model
model = eval(args.model)(
        max_n=graph_args.max_n, 
        nvt=graph_args.nvt,
        net=graph_args.net,
        START_TYPE=graph_args.START_TYPE, 
        END_TYPE=graph_args.END_TYPE, 
        hs=args.hs, 
        nz=args.nz
        )
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

'''
# plot sample train/test graphs
if not (os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.pdf')) or os.path.exists(os.path.join(args.res_dir, 'train_graph_id0.png'))):
    for data in ['train_data', 'test_data']:
        G = [g for g, y in eval(data)[:10]]
        for i, g in enumerate(G):
            name = '{}_graph_id{}'.format(data[:-5], i)
            plot_DAG(g, args.res_dir, name)
'''

'''Define some train/test functions'''
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss   = 0
    pred_loss  = 0
    pbar       = tqdm(train_data)
    g_batch    = []
    y_batch    = []
    for i, (g, y) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            '''
            mu, logvar = model.encode(g_batch)
            loss, recon, kld = model.loss(mu, logvar, g_batch)
            '''
            loss, recon, kld = model(g_batch)
            pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), kld.item()/len(g_batch)))
            loss.backward()
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            optimizer.step()
            g_batch = []
            y_batch = []
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_data)))
    return train_loss, recon_loss, kld_loss

def test():
    # test recon accuracy
    test_model.eval()
    encode_times = 1
    decode_times = 1
    Nll = 0
    n_perfect = 0
    print('Testing begins...')

    print('Performence on the train data: ')
    pbar1 = tqdm(train_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar1):
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(train_data) - 1:
            g = test_model._collate_fn(g_batch)
            mu, logvar = test_model.encode(g)
            _, nll, _ = test_model.loss(mu, logvar, g)
            pbar1.set_description('recon loss: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            # construct igraph g from tensor g to check recon quality
            for _ in range(encode_times):
                z = test_model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    g_recon = test_model.decode(z)
                    n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
            y_batch = []
    Nll /= len(train_data)
    acc = n_perfect / (len(train_data) * encode_times * decode_times)
    print('Trainset average recon loss: {0}, recon accuracy: {1:.4f}'.format(Nll, acc))

    print('Performence on the test data: ')
    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    Nll = 0
    n_perfect = 0
    for i, (g, y) in enumerate(pbar):
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = test_model._collate_fn(g_batch)
            mu, logvar = test_model.encode(g)
            _, nll, _ = test_model.loss(mu, logvar, g)
            pbar.set_description('recon loss: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            # construct igraph g from tensor g to check recon quality
            for _ in range(encode_times):
                z = test_model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    g_recon = test_model.decode(z)
                    n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
            y_batch = []
    Nll /= len(test_data)
    acc = n_perfect / (len(test_data) * encode_times * decode_times)
    print('Testset average recon loss: {0}, recon accuracy: {1:.4f}'.format(Nll, acc))
    #return Nll, acc

def visualize_recon(epoch, current_model):
    current_model.eval()
    # draw some reconstructed train/test graphs to visualize recon quality
    for i, (g, y) in enumerate(test_data[:10]+train_data[:10]):
        g_recon = current_model.encode_decode(g)[0]
        name0 = 'graph_epoch{}_id{}_original'.format(epoch, i)
        plot_DAG(g, args.res_dir, name0)
        name1 = 'graph_epoch{}_id{}_recon'.format(epoch, i)
        plot_DAG(g_recon, args.res_dir, name1)

def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y) in enumerate(tqdm(data)):
        # copy igraph
        # otherwise original igraphs will save the H states and consume more GPU memory
        g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)

def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )

def interpolation_exp(current_model, epoch, num=3):
    print('Interpolation experiments between two random testing graphs')
    interpolation_res_dir = os.path.join(args.res_dir, 'interpolation')
    if not os.path.exists(interpolation_res_dir):
        os.makedirs(interpolation_res_dir) 
    interpolate_number = 10
    current_model.eval()
    cnt = 0
    for i in range(0, len(test_data), 2):
        cnt += 1
        (g0, _), (g1, _) = test_data[i], test_data[i+1]
        z0, _ = current_model.encode(g0)
        z1, _ = current_model.encode(g1)
        print('norm of z0: {}, norm of z1: {}'.format(torch.norm(z0), torch.norm(z1)))
        print('distance between z0 and z1: {}'.format(torch.norm(z0-z1)))
        Z = []  # to store all the interpolation points
        for j in range(0, interpolate_number + 1):
            zj = z0 + (z1 - z0) / interpolate_number * j
            Z.append(zj)
        Z = torch.cat(Z, 0)
        # decode many times and select the most common one
        G, G_str = decode_from_latent_space(Z, current_model, return_igraph=True) 
        names = []
        scores = []
        for j in range(0, interpolate_number + 1):
            namej = 'graph_interpolate_{}_{}_of_{}'.format(i, j, interpolate_number)
            namej = plot_DAG(G[j], interpolation_res_dir, namej, backbone=True)
            names.append(namej)
        fig = plt.figure(figsize=(120, 20))
        for j, namej in enumerate(names):
            imgj = mpimg.imread(namej)
            fig.add_subplot(1, interpolate_number + 1, j + 1)
            plt.imshow(imgj)
            plt.axis('off')
        plt.savefig(os.path.join(args.res_dir, 
                    args.data_name + '_{}_interpolate_exp_ensemble_epoch{}_{}.pdf'.format(
                    args.model, epoch, i)), bbox_inches='tight')

        if cnt == num:
            break

def smoothness_exp(current_model, epoch, gap=0.05):
    print('Smoothness experiments around a latent vector')
    smoothness_res_dir = os.path.join(args.res_dir, 'smoothness')
    if not os.path.exists(smoothness_res_dir):
        os.makedirs(smoothness_res_dir) 
    
    #z0 = torch.zeros(1, model.nz).to(device)  # use all-zero vector as center
    row = [[0], [1, 5], [1, 8, 6], [2, 7, 14, 5], [3, 0, 2, 4, 0]] # a example opamp circuit
    #row = flat_CIRCUIT_to_nested(row, model.max_n-2)
    g0, _ = decode_CIRCUIT_to_igraph(row)
    z0, _ = current_model.encode(g0)

    # select two orthogonal directions in latent space
    tmp = np.random.randn(z0.shape[1], z0.shape[1])
    Q, R = qr(tmp)
    dir1 = torch.FloatTensor(tmp[0:1, :]).to(device)
    dir2 = torch.FloatTensor(tmp[1:2, :]).to(device)

    # generate architectures along two orthogonal directions
    grid_size = 13
    grid_size = 9
    mid = grid_size // 2
    Z = []
    pbar = tqdm(range(grid_size ** 2))
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        zij = z0 + dir1 * (i - mid) * gap + dir2 * (j - mid) * gap
        Z.append(zij)
    Z = torch.cat(Z, 0)
    if True:
        G, _ = decode_from_latent_space(Z, current_model, return_igraph=True)
    else:  # decode by 3 batches in case of GPU out of memory 
        Z0, Z1, Z2 = Z[:len(Z)//3, :], Z[len(Z)//3:len(Z)//3*2, :], Z[len(Z)//3*2:, :]
        G = []
        G += decode_from_latent_space(Z0, current_model, return_igraph=True)[0]
        G += decode_from_latent_space(Z1, current_model, return_igraph=True)[0]
        G += decode_from_latent_space(Z2, current_model, return_igraph=True)[0]
    names = []
    for idx in pbar:
        i, j = divmod(idx, grid_size)
        pbar.set_description('Drawing row {}/{}, col {}/{}...'.format(i+1, 
                             grid_size, j+1, grid_size))
        nameij = 'graph_smoothness{}_{}'.format(i, j)
        nameij = plot_DAG(G[idx], smoothness_res_dir, nameij)
        names.append(nameij)

    fig = plt.figure(figsize=(50, 50))
    
    nrow, ncol = grid_size, grid_size
    for ij, nameij in enumerate(names):
        imgij = mpimg.imread(nameij)
        fig.add_subplot(nrow, ncol, ij + 1)
        plt.imshow(imgij)
        plt.axis('off')
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.savefig(os.path.join(args.res_dir, 
                args.data_name + '_{}_smoothness_ensemble_epoch{}_gap={}_small.pdf'.format(
                args.model, epoch, gap)), bbox_inches='tight')

def distance_exp(current_model, num=100):
    NZ0 = 0
    NZ1 = 0
    NZ2 = 0
    D1  = 0
    D2  = 0
    for i in range(num):
        v0    = sample_topo_vector(CIRCUIT_NODE_NUM)
        v1    = list(np.round(np.array(v0) + random.random()))
        v2    = sample_topo_vector(CIRCUIT_NODE_NUM)
        g0, _ = decode_CIRCUIT_to_igraph(vector2row(v0))
        g1, _ = decode_CIRCUIT_to_igraph(vector2row(v1))
        g2, _ = decode_CIRCUIT_to_igraph(vector2row(v2))
        z0, _ = current_model.encode(g0)
        z1, _ = current_model.encode(g1)
        z2, _ = current_model.encode(g2)
        nz0   = torch.norm(z0)
        nz1   = torch.norm(z1)
        nz2   = torch.norm(z2)
        d1    = torch.norm(z0-z1)
        d2    = torch.norm(z0-z2)
        NZ0   = NZ0 + nz0
        NZ1   = NZ1 + nz1
        NZ2   = NZ2 + nz2
        D1    = D1  + d1
        D2    = D2  + d2
    print('average norm of# z0: {}, z1: {}, z2: {}'.format(NZ0/num, NZ1/num, NZ2/num))
    print('average distance between z0 and z1: {}'.format(D1/num))
    print('average distance between z0 and z2: {}'.format(D2/num))

'''Training begins here'''
if args.only_test:
    '''Only testing'''
    load_model_path  = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, args.load_model_path))
    load_model_name = 'model_checkpoint' + args.load_model_name + '.pth'
    test_model = torch.load(os.path.join(load_model_path, load_model_name))
    print('model: {} has been loaded'.format(os.path.join(load_model_path, load_model_name)))
    visualize_recon(args.epochs, test_model)
    test()
    print('begin distance testing...')
    distance_exp(test_model, num=100)
    '''
    print('begin interpolation testing...')
    interpolation_exp(test_model, args.epochs, num=3)
    print('begin smoothness testing...')
    smoothness_exp(test_model, args.epochs)
    '''
else:
    min_loss = math.inf
    min_loss_epoch = None
    loss_name = os.path.join(args.res_dir, 'train_loss.txt')
    loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
    test_results_name = os.path.join(args.res_dir, 'test_results.txt')

    if os.path.exists(loss_name):
        os.remove(loss_name)
    for epoch in range(1, args.epochs + 1):
        train_loss, recon_loss, kld_loss = train(epoch)
        with open(loss_name, 'a') as loss_file:
            loss_file.write("{:.2f} {:.2f} {:.2f}\n".format(
                train_loss/len(train_data), 
                recon_loss/len(train_data), 
                kld_loss/len(train_data), 
                ))
        scheduler.step(train_loss)
        if epoch % args.save_interval == 0:
            print("save current model...")
            model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
            optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
            scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
            #torch.save(model.state_dict(), model_name)
            torch.save(model, model_name)
            torch.save(optimizer.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)
            print("visualize reconstruction examples...")
            visualize_recon(epoch, model)
            #print("extract latent representations...")
            #save_latent_representations(epoch)
            print("sample from prior...")
            sampled = model.generate_sample(args.sample_number)
            for i, g in enumerate(sampled):
                namei = 'graph_{}_sample{}'.format(epoch, i)
                plot_DAG(g, args.res_dir, namei)
            print("plot train loss...")
            losses = np.loadtxt(loss_name)
            if losses.ndim == 1:
                continue
            fig = plt.figure()
            num_points = losses.shape[0]
            plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
            plt.plot(range(1, num_points+1), losses[:, 1], label='Recon')
            plt.plot(range(1, num_points+1), losses[:, 2], label='KLD')
            plt.xlabel('Epoch')
            plt.ylabel('Train loss')
            plt.legend()
            plt.savefig(loss_plot_name)


