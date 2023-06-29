import math
import random
import torch
from torch import nn
from torch.nn import functional as F, MSELoss as MSE
import torch.nn.init as init
import numpy as np
import igraph
import pdb


class DVAE_hybirdLoss(nn.Module):
    def __init__(self, max_n, nvt, net, START_TYPE, END_TYPE, hs=501, nz=56, vid=True):
        super(DVAE_hybirdLoss, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.net = net  # number of edge types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.device = None
        self.vid = vid # whether use the vertex id of graph
        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRUCell(nvt, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 8), 
                nn.ReLU(), 
                nn.Linear(hs * 8, net)
                )  # which type edge to add between v_i and v_new, f(hvi, hnew)
                   # add one linear layer here

        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros

        # 3. bidir-related, to unify sizes, ignored

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
    
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        H_pred = []
        H_name = 'H_forward'

        for g in G:
            tmp = []
            for x in g.predecessors(v):
                eid = g.get_eid(x,v)
                _edge_type = g.es['weight'][eid]
                tmp.append((g.es['weight'][eid]*10+1)*g.vs[x][H_name]) # add the edge type info, edge type * g[H_name]
                '''
                if _edge_type != 0:
                    tmp.append(g.es['weight'][eid]*g.vs[x][H_name]) # add the edge type info
                else:
                    tmp.append(g.vs[x][H_name])
                '''
            H_pred.append(tmp)

        if self.vid:
            vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]

        gate, mapper = self.gate_forward, self.mapper_forward

        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)

        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0)
        return
    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)))
        Hg = self._get_graph_state(G)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu
        #std = logvar.mul(0.5).exp_()
        #eps = torch.randn_like(std) * eps_scale
        #return eps.mul(std).add_(mu)

    def _get_edge_type(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return torch.argmax(self.add_edge(torch.cat([Hvi, H], -1)), 1)

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.

        # init the vertex status
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)     
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # the order of vertex is fixed
            if idx == self.max_n - 1:
                new_types = [self.END_TYPE] * len(G)
            elif idx <= 2:
                new_types = [idx+1] * len(G)
            else:
                new_types = [idx] * len(G)
            for i, g in enumerate(G):
                if not finished[i]:
                    g.add_vertex(type=new_types[i])
            #self._update_v(G, idx)
        for g in G:
            g.vs[0]['color'] ="pink"
            g.vs[1]['color'] = 'black'
            g.vs[2]['color'] = 'yellow'
            g.vs[3]['color'] = 'yellow'
            g.vs[4]['color'] = 'blue'
            g.vs[5]['color'] = 'red'
            g.vs[6]['color'] ="pink"
            g.add_edge(0, 1, weight=0) #connect virtual start node and Vin
            g.add_edge(4, 6, weight=0) #connect virtual end node and Vout
            g.add_edge(5, 6, weight=0) #connect virtual end node and GND
            #g.add_edge(1, 2, weight=5)
            #g.add_edge(2, 3, weight=6)
            #g.add_edge(3, 4, weight=5) # fix 5 connections
            #g.add_edge(1, 5, weight=0)
            #g.add_edge(4, 5, weight=0)
        for v in range(0, self.max_n):
            self._update_v(G, v)

        # decide connection type
        for idx in range(2, self.max_n-1):
            for vi in range(idx-1, 0, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, idx)
                #pred_type = self._get_edge_type(Hvi, H, H0).cpu().numpy().tolist()
                ei_probs = self.add_edge(torch.cat([Hvi, H], -1)).float()
                type_probs = F.softmax(ei_probs, 1).cpu().detach().numpy()
                pred_type = [np.random.choice(range(self.net), p=type_probs[i]) for i in range(len(G))]
                for i, g in enumerate(G):
                    g.add_edge(vi, idx, weight=int(pred_type[i]))
                self._update_v(G, idx)
        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct

        # init the vertex status
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)  
        for v_true in range(1, self.max_n):
            # the order of vertex is fixed
            if v_true == self.max_n - 1:
                new_types = [self.END_TYPE] * len(G)
            elif v_true <= 2:
                new_types = [v_true+1] * len(G)
            else:
                new_types = [v_true] * len(G) # include START TYPE
            for i, g in enumerate(G):
                    g.add_vertex(type=new_types[i])
            #self._update_v(G, v_true)
        for g in G:
            g.add_edge(0, 1, weight=0) # connect virtual start node and Vin
            g.add_edge(4, 6, weight=0) # connect virtual end node and Vout
            g.add_edge(5, 6, weight=0) # connect virtual end node and GND
            #g.add_edge(1, 2, weight=5)
            #g.add_edge(2, 3, weight=6)
            #g.add_edge(3, 4, weight=5) # fix 5 connections
            #g.add_edge(1, 5, weight=0)
            #g.add_edge(4, 5, weight=0)
        for v in range(0, self.max_n):
            self._update_v(G, v)

        # calculate the likelihood of adding true types of edges
        res = 0  # log likelihood
        fix_e = [[1,2],[2,3],[3,4],[1,5],[4,5]]
        for v_true in range(2,self.max_n-1):
            #print(v_true)
            true_edge_types_tmp = []
            true_edge_types = []
            pred_edge_types = []
            true_edge_types_oh = []
            pred_edge_types_oh = []
            ell1 = 0
            for vi in range(v_true-1, 0, -1):
                Hvi = self._get_vertex_state(G, vi)
                H   = self._get_vertex_state(G, v_true)
                pred_type = self._get_edge_type(Hvi, H, H0).cpu().numpy().tolist()
                #ei_probs = self.add_edge(torch.cat([Hvi, H], -1)).float()
                #type_probs = F.softmax(ei_probs, 1).cpu().detach().numpy()
                #pred_type = [np.random.choice(range(self.net), p=type_probs[i]) for i in range(len(G))]
                edge_type = []
                for i, g in enumerate(G):
                    g_true = G_true[i]
                    _eid = g_true.get_eid(vi,v_true)
                    _edge_type = g_true.es['weight'][_eid]
                    g.add_edge(vi, v_true, weight=int(_edge_type))
                    edge_type.append(_edge_type)
                true_edge_types_tmp.append(edge_type)
                ei_probs = self.add_edge(torch.cat([Hvi, H], -1)).float()
                ell = self.logsoftmax1(ei_probs)[np.arange(len(G)), true_edge_types_tmp].sum()
                ell1 = ell1 + ell
                edge_type_oh = self._one_hot(edge_type, self.net).T.cpu().numpy().tolist()
                pred_type_oh = self._one_hot(pred_type, self.net).T.cpu().numpy().tolist()
                pred_edge_types.append(pred_type)
                true_edge_types.append(edge_type)
                pred_edge_types_oh.append(pred_type_oh)
                true_edge_types_oh.append(edge_type_oh)
                self._update_v(G, v_true)
            pred_edge_types = torch.tensor(pred_edge_types)
            true_edge_types = torch.tensor(true_edge_types)
            pred_edge_types_oh = torch.tensor(pred_edge_types_oh)
            true_edge_types_oh = torch.tensor(true_edge_types_oh)

            # edges log-likelihood
            loss_func = MSE(reduction='mean')
            ell3 = - loss_func(pred_edge_types.float(), true_edge_types.float())
            ell2 = - F.binary_cross_entropy(pred_edge_types_oh, true_edge_types_oh, reduction='sum')
            #res = res + ell1 + ell2 + ell3*0.2
            res = res + ell1
        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, recon, kld = self.loss(mu, logvar, G)
        return loss, recon, kld
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G


'''
    A fast D-VAE variant.
    Use D-VAE's encoder + a simple decoder to accelerate decoding.
'''

class DVAE_test2(DVAE_hybirdLoss):
    def __init__(self, max_n, nvt, net, START_TYPE, END_TYPE, hs=501, nz=56, vid=True):
        super(DVAE_test2, self).__init__(max_n, nvt, net, START_TYPE, END_TYPE, hs, nz, vid)
        self.grud = nn.GRU(hs, hs, batch_first=True)  # decoder GRU
        self.add_edges = nn.Sequential(
                nn.Linear(hs, hs*4), 
                nn.ReLU(), 
                nn.Linear(hs*4, hs*2), 
                nn.ReLU(), 
                nn.Linear(hs*2, self.net)
                )  # which type edge to add between v_i and v_new, f(hvi, hnew)
                   # add one linear layer here

    def _decode(self, z):
        h0 = self.relu(self.fc3(z))
        h_in = h0.unsqueeze(1).expand(-1, 10, -1)
        h_out, _ = self.grud(h_in)
        edge_scores = self.sigmoid(self.add_edges(h_out))  # batch * max_n-1 * self.net
        return edge_scores.permute(1,2,0)
        #return edge_scores

    def loss(self, mu, logvar, G_true, beta=0.005, ml=True):
        true_edge_types = []
        true_edge_types_oh = []
        for v_true in range(2,self.max_n-1):
            for vi in range(v_true-1, 0, -1):
                edge_type = []
                for i, g_true in enumerate(G_true):
                    _eid = g_true.get_eid(vi,v_true)
                    _edge_type = g_true.es['weight'][_eid]
                    edge_type.append(_edge_type)
                true_edge_types.append(edge_type)
                edge_type_oh = self._one_hot(edge_type, self.net).T.cpu().numpy().tolist()
                true_edge_types_oh.append(edge_type_oh)
        true_edge_types = torch.tensor(true_edge_types).to(self.get_device())
        true_edge_types_oh = torch.tensor(true_edge_types_oh).to(self.get_device())

        z = self.reparameterize(mu, logvar)
        edge_scores = self._decode(z)
        res = 0
        res += F.binary_cross_entropy(edge_scores, true_edge_types_oh, reduction='sum')

        if ml:
            pred_edges  = self.get_pred_edges(edge_scores)
            true_edge_types = true_edge_types.transpose(1, 0)
            loss_func = MSE(reduction='mean')
            res = res + loss_func(pred_edges.float(), true_edge_types.float()) * 0.5

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def decode(self, z):
        edge_scores = self._decode(z)
        return self.construct_igraph(edge_scores)
    
    def get_pred_edges(self, edge_scores, stochastic=False):
        pred_edges  = np.zeros((edge_scores.shape[2],10))
        G = [igraph.Graph(directed=True) for _ in range(edge_scores.shape[2])]
        for i,g in enumerate(G):
            g.add_vertex(type=self.START_TYPE)
            for idx in range(1, self.max_n):
                # the order of vertex is fixed
                if idx == self.max_n - 1:
                    new_types = [self.END_TYPE] * len(G)
                elif idx <= 2:
                    new_types = [idx+1] * len(G)
                else:
                    new_types = [idx] * len(G)
                g.add_vertex(type=new_types[i])

            g.vs[0]['color'] ="pink"
            g.vs[1]['color'] = 'black'
            g.vs[2]['color'] = 'yellow'
            g.vs[3]['color'] = 'yellow'
            g.vs[4]['color'] = 'blue'
            g.vs[5]['color'] = 'red'
            g.vs[6]['color'] ="pink"

            j = 0

            for m in range(2,self.max_n-1):
                for n in range(m-1,0,-1):
                    g.add_edge(n,m)
            g.add_edge(0, 1, weight=0) #connect virtual start node and Vin
            g.add_edge(4, 6, weight=0) #connect virtual end node and Vout
            g.add_edge(5, 6, weight=0) #connect virtual end node and GND

            for idx in range(2, self.max_n-1):
                for vi in range(idx-1, 0, -1):
                    _eid = g.get_eid(vi,idx)
                    if stochastic:
                        type_probs = F.softmax(edge_scores, 1).cpu().detach().numpy()
                        pred_type = [np.random.choice(range(self.net), p=type_probs[_eid,:,i])]
                    else:
                        pred_type = torch.argmax(edge_scores[_eid,:,i], 0).cpu().numpy().tolist()
                    g.es[_eid]['weight'] = int(pred_type)
                    #g.add_edge(vi, idx, weight=int(pred_type[0]))

                    pred_edges[i][j] = int(pred_type)
                    j = j + 1

        return torch.tensor(pred_edges).to(self.get_device())

    def construct_igraph(self, edge_scores, stochastic=False):

        G = [igraph.Graph(directed=True) for _ in range(edge_scores.shape[2])]
        for i,g in enumerate(G):
            g.add_vertex(type=self.START_TYPE)
            for idx in range(1, self.max_n):
                # the order of vertex is fixed
                if idx == self.max_n - 1:
                    new_types = [self.END_TYPE] * len(G)
                elif idx <= 2:
                    new_types = [idx+1] * len(G)
                else:
                    new_types = [idx] * len(G)
                g.add_vertex(type=new_types[i])

            g.vs[0]['color'] ="pink"
            g.vs[1]['color'] = 'black'
            g.vs[2]['color'] = 'yellow'
            g.vs[3]['color'] = 'yellow'
            g.vs[4]['color'] = 'blue'
            g.vs[5]['color'] = 'red'
            g.vs[6]['color'] ="pink"
            for m in range(2,self.max_n-1):
                for n in range(m-1,0,-1):
                    g.add_edge(n,m)
            g.add_edge(0, 1, weight=0) #connect virtual start node and Vin
            g.add_edge(4, 6, weight=0) #connect virtual end node and Vout
            g.add_edge(5, 6, weight=0) #connect virtual end node and GND

            for idx in range(2, self.max_n-1):
                for vi in range(idx-1, 0, -1):
                    _eid = g.get_eid(vi,idx)
                    if stochastic:
                        type_probs = F.softmax(edge_scores, 1).cpu().detach().numpy()
                        pred_type = [np.random.choice(range(self.net), p=type_probs[_eid,:,i])]
                    else:
                        pred_type = torch.argmax(edge_scores[_eid,:,i], 0).cpu().numpy().tolist()
                    g.es[_eid]['weight'] = int(pred_type)
                    #g.add_edge(vi, idx, weight=int(pred_type[0]))

        return G
