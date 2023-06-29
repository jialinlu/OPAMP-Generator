import gzip
import pickle
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
import os
import subprocess
import collections
import igraph
import argparse
import pdb
import pygraphviz as pgv
import sys
from PIL import Image

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

'''gobal variables'''
CIRCUIT_NODE_NUM  = 5
CIRCUIT_NODE_TYPE = 4
CIRCUIT_EDGE_NUM  = 10 # fix 5, tune 5, total 10
CIRCUIT_EDGE_TYPE = 25

'''dataset generation'''

def clamp(minn, maxn, n):
    return max(min(maxn, n), minn)

def sample_topo_vector(edge_num):
    topo_vector = []
    for e in range(edge_num):
        if e<=1:
            topo_vector.append(np.random.randint(0,11))
        elif e==2:
            topo_vector.append(np.random.randint(0,25))
        else:
            topo_vector.append(np.random.randint(0,5))
    return topo_vector

def sample_full_random(edge_num):
    topo_vector = []
    for e in range(edge_num):
        topo_vector.append(np.random.randint(0,25))
    return topo_vector

def vector2row(topo_vector):
    CIRCUIT_DAG = []
    for i in range(CIRCUIT_NODE_NUM):
        if i == 0:
            CIRCUIT_DAG.append([0]) # node Vin
        elif i == 1:
            CIRCUIT_DAG.append([1,5]) # node 1
        elif i == 2:
            CIRCUIT_DAG.append([1,topo_vector[0],6]) # node 2
        elif i == 3:
            CIRCUIT_DAG.append([2,topo_vector[1],topo_vector[2],5]) # node Vout
        elif i == 4:
            CIRCUIT_DAG.append([3,0,topo_vector[3],topo_vector[4],0]) # node GND
        else:
            pass
    return CIRCUIT_DAG

def row2vector(row):
    topo = []
    for i in range(len(row)):
        if i == 2:
            topo.append(row[i][1])
        elif i == 3:
            topo.append(row[i][1])
            topo.append(row[i][2])
        elif i == 4:
            topo.append(row[i][2])
            topo.append(row[i][3])
        else:
            pass
    return np.array(topo)

def g2topo(g):
    #plot_DAG(g, './', 'fuck')
    row_ = []
    for vo in range(2, g.vcount()-1):
        for vi in range(1,vo):
            _eid = g.get_eid(vi,vo)
            row_.append(int(g.es[_eid]['weight']))
    topo_id = [1,3,4,7,8]
    topo = []
    for i in topo_id:
        if i==1 or i==3:
            topo.append(clamp(0,11,row_[i]))
        elif i ==4:
            topo.append(clamp(0,25,row_[i]))
        else:
            topo.append(clamp(0,5,row_[i]))
    return np.array(topo)

'''Data preprocessing'''
def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x

def load_CIRCUIT_graphs(name, rand_seed=0, with_y=False):
    # load DAG format CIRCUITs to igraphs
    g_list = []
    max_n = 0  # maximum number of nodes
    with open('./%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            g, n = decode_CIRCUIT_to_igraph(row)
            max_n = max(max_n, n)
            g_list.append((g, y)) 
    graph_args.nvt = CIRCUIT_NODE_TYPE + 2 # original types + virtual start/end types
    graph_args.net = CIRCUIT_EDGE_TYPE     # edge weight in igraph
    graph_args.max_n = max_n               # maximum number of nodes
    graph_args.START_TYPE = 1              # predefined start vertex type
    graph_args.END_TYPE = 0                # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.nvt)
    print('# edge types: %d' % graph_args.net)
    #random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args

def decode_CIRCUIT_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    #g = igraph.Graph(directed=False) # use undirected graph
    g.add_vertices(n+2)
    g.vs[0]['type'] = 0  # virtual start node
    g.vs[0]['color'] ="pink"
    g.vs[1]['type'] = 2 # Vin, 0+2
    g.vs[1]['color'] = 'black'
    g.vs[2]['type'] = 3 # node 1, 1+2
    g.vs[2]['color'] = 'yellow'
    g.vs[3]['type'] = 3 # node 2, 1+2
    g.vs[3]['color'] = 'yellow'
    g.vs[4]['type'] = 4 # Vout, 2+2
    g.vs[4]['color'] = 'blue'
    g.vs[5]['type'] = 5 # GND, 3+2
    g.vs[5]['color'] = 'red'
    g.vs[6]['type'] = 1  # virtual end node
    g.vs[6]['color'] ="pink"
    edge_pairs = [[1,2],[1,3],[2,3],[1,4],[2,4],[3,4],[1,5],[2,5],[3,5],[4,5]]
    edge_types = [[1,1],[2,1],[2,2],[3,1],[3,2],[3,3],[4,1],[4,2],[4,3],[4,4]]
    for i,ep in enumerate(edge_pairs):
        edge_type  = row[edge_types[i][0]][edge_types[i][1]]
        g.add_edge(ep[0],ep[1], weight=edge_type)
    g.add_edge(0, 1, weight=0) #connect virtual start node and Vin
    g.add_edge(4, 6, weight=0) #connect virtual end node and Vout
    g.add_edge(5, 6, weight=0) #connect virtual end node and GND
    return g, n+2

'''Network visualization'''
def plot_DAG(g, res_dir, name, backbone=False, pdf=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name+'.png')
    if pdf:
        file_name = os.path.join(res_dir, name+'.pdf')
    draw_network(g, file_name, backbone)
    return file_name

def draw_network(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    # add vertex
    for idx in range(1,g.vcount()-1):
        add_node(graph, idx, g.vs[idx]['type'])
    # add edge
    for idx in range(2,g.vcount()-1):
        for node in g.get_adjlist(igraph.IN)[idx]:
            eid = g.get_eid(node,idx)
            edge_label = g.es['weight'][eid]
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1, label=edge_label)
            else:
                graph.add_edge(node, idx, weight=0, label=edge_label)
    graph.layout(prog='dot')
    graph.draw(path)

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 2:
        label = 'Vin'
        color = 'pink'
    elif label == 3:
        if node_id == 2:
            label = 'node1'
            color = 'yellow'
        else:
            label = 'node2'
            color = 'yellow'
    elif label == 4:
        label = 'Vout'
        color = 'greenyellow'
    elif label == 5:
        label = 'GND'
        color = 'seagreen3'
    else:
        label = 'test'
        color = 'aliceblue'
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)

'''Validity and novelty functions'''
def is_same_DAG(g0, g1):
    # Correct rate of edge type prediction
    all_e = 0
    right_e = 0
    #fix_e = [[1,2],[2,3],[3,4],[1,5],[4,5]]
    for vo in range(2, g0.vcount()-1):
        for vi in g0.get_adjlist(igraph.IN)[vo]:
            eid0 = g0.get_eid(vi,vo)
            eid1 = g1.get_eid(vi,vo)
            g0_edge_type = g0.es['weight'][eid0]
            g1_edge_type = g1.es['weight'][eid1]
            if g0_edge_type == g1_edge_type:
                right_e = right_e + 1
            all_e = all_e + 1
    return right_e / all_e

def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

def decode_igraph_to_CIRCUIT(g):
    # decode an igraph to a flattend CIRCUIT string
    n = g.vcount()
    res = []
    adjlist = g.get_adjlist(igraph.IN)
    for i in range(1, n-1):
        res.append(int(g.vs[i]['type'])-2)
        row = [0] * (i-1)
        for j in adjlist[i]:
            if j < i-1:
                row[j] = 1
        res += row
    return ' '.join(str(x) for x in res)
    #return res

def decode_from_latent_space(
        latent_points, model, decode_attempts=500, n_nodes='variable', return_igraph=False):
    # decode points from the VAE model's latent space multiple attempts
    # and return the most common decoded graphs
    if n_nodes != 'variable':
        check_n_nodes = True  # check whether the decoded graphs have exactly n nodes
    else:
        check_n_nodes = False
    decoded_arcs = []  # a list of lists of igraphs
    pbar = tqdm(range(decode_attempts))
    for i in pbar:
        current_decoded_arcs = model.decode(latent_points)
        decoded_arcs.append(current_decoded_arcs)
        pbar.set_description("Decoding attempts {}/{}".format(i, decode_attempts))

    # We see which ones are decoded to be valid architectures
    valid_arcs = []  # a list of lists of strings
    if return_igraph:
        str2igraph = {}  # map strings to igraphs
    pbar = tqdm(range(latent_points.shape[0]))
    for i in pbar:
        valid_arcs.append([])
        for j in range(decode_attempts):
            arc = decoded_arcs[j][i]  # arc is an igraph
            if not check_n_nodes or check_n_nodes and arc.vcount() == n_nodes:
                cur = decode_igraph_to_CIRCUIT(arc)  # a flat circuit igraph string
                if return_igraph:
                    str2igraph[cur] = arc
                valid_arcs[i].append(cur)
        pbar.set_description("Check validity for {}/{}".format(i, latent_points.shape[0]))

    # select the most common decoding as the final architecture
    final_arcs = []  # a list of lists of strings
    pbar = tqdm(range(latent_points.shape[ 0 ]))
    for i in pbar:
        valid_curs = valid_arcs[i]
        aux = collections.Counter(valid_curs)
        if len(aux) > 0:
            arc, num_arc = list(aux.items())[np.argmax(aux.values())]
        else:
            arc = None
            num_arc = 0
        final_arcs.append(arc)
        pbar.set_description("Latent point {}'s most common decoding ratio: {}/{}".format(
                             i, num_arc, len(valid_curs)))

    if return_igraph:
        final_arcs_igraph = [str2igraph[x] if x is not None else None for x in final_arcs]
        return final_arcs_igraph, final_arcs
    return final_arcs
