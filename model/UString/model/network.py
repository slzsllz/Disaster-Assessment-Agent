"""Inference layers adapted from UString, ACM MM 2020 (see LICENSE)."""
import inspect
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from .bayesian import BayesianLinear

def glorot(tensor):
    if tensor is not None:
        nn.init.xavier_uniform_(tensor)

def zeros(tensor):
    if tensor is not None:
        nn.init.zeros_(tensor)

def scatter_add(src, index, dim=0, dim_size=None):
    shape = list(src.shape)
    shape[dim] = dim_size if dim_size is not None else int(index.max()) + 1
    return src.new_zeros(shape).index_add_(dim, index, src)

def scatter_(name, src, index, dim_size=None):
    if name != 'add':
        raise ValueError('UString inference only uses additive aggregation')
    return scatter_add(src, index, dim_size=dim_size)

class MessagePassing(torch.nn.Module):

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getfullargspec(self.message)[0][1:]
        self.update_args = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index
        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])
        update_args = [kwargs[arg] for arg in self.update_args]
        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class GCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def add_self_loops(self, edge_index, edge_weight=None, fill_value=1, num_nodes=None):
        batch_size = edge_index.size(0)
        num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_index = loop_index.unsqueeze(0).repeat(batch_size, 1, 1)
        if edge_weight is not None:
            assert edge_weight.size(-1) == edge_index.size(-1)
            loop_weight = edge_weight.new_full((num_nodes,), fill_value)
            loop_weight = loop_weight.unsqueeze(0).repeat(batch_size, 1)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        return (edge_index, edge_weight)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(0), edge_index.size(-1)), dtype=x.dtype, device=x.device)
        assert edge_weight.size(-1) == edge_index.size(-1)
        edge_index, edge_weight = self.add_self_loops(edge_index, edge_weight=edge_weight, num_nodes=x.size(1))
        out_batch = []
        for i in range(edge_index.size(0)):
            row, col = edge_index[i]
            deg = scatter_add(edge_weight[i], row, dim=0, dim_size=x.size(1))
            deg_inv = deg.pow(-0.5)
            deg_inv[deg_inv == float('inf')] = 0
            norm = deg_inv[row] * edge_weight[i] * deg_inv[col]
            weight = self.weight.to(x.device)
            x_w = torch.matmul(x[i], weight)
            out = self.propagate('add', edge_index[i], x=x_w, norm=norm)
            out_batch.append(self.act(out))
        out_batch = torch.stack(out_batch)
        return out_batch

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            bias = self.bias.to(aggr_out.device)
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class Graph_GRU_GCN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(Graph_GRU_GCN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = nn.ModuleList()
        self.weight_hz = nn.ModuleList()
        self.weight_xr = nn.ModuleList()
        self.weight_hr = nn.ModuleList()
        self.weight_xh = nn.ModuleList()
        self.weight_hh = nn.ModuleList()
        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h, edge_weight=None):
        h_out = torch.zeros(h.size())
        h_out = h_out.to(inp.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        return h_out

class BayesianPredictor(nn.Module):

    def __init__(self, input_dim, output_dim=2, act=torch.relu, pi=0.5, sigma_1=None, sigma_2=None):
        super(BayesianPredictor, self).__init__()
        self.act = act
        self.l1 = BayesianLinear(input_dim, 64, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)
        self.l2 = BayesianLinear(64, output_dim, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)

    def forward(self, x, sample=False):
        x = self.act(self.l1(x, sample))
        x = self.l2(x, sample)
        return x

class UString(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True, uncertain_ranking=False):
        super(UString, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa
        self.uncertain_ranking = uncertain_ranking
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.enc_gcn1 = GCNConv(h_dim + h_dim, h_dim)
        self.enc_gcn2 = GCNConv(h_dim + h_dim, z_dim, act=lambda x: x)
        self.rnn = Graph_GRU_GCN(h_dim + h_dim + z_dim, h_dim, n_layers, bias=True)
        self.predictor = BayesianPredictor(n_obj * z_dim, 2)

    def forward(self, features, edges, weights, samples=10):
        hidden = features.new_zeros((self.n_layers, 1, self.n_obj, self.h_dim))
        outputs = []
        graph = edges.unsqueeze(0)
        for frame, frame_weights in zip(features, weights):
            embedded = self.phi_x(frame).unsqueeze(0)
            x = torch.cat((embedded[:, 1:], embedded[:, :1].expand(-1, self.n_obj, -1)), -1)
            w = frame_weights.unsqueeze(0)
            enc = self.enc_gcn1(x, graph, w)
            latent = self.enc_gcn2(torch.cat((enc, hidden[-1]), -1), graph, w)
            logits = torch.stack([self.predictor(latent.flatten(1), sample=True) for _ in range(samples)])
            p = logits.softmax(-1)
            score = logits.mean(0).softmax(-1)[0, 1]
            aleatoric = (p * (1 - p)).sum(-1).mean()
            epistemic = ((p - p.mean(0)) ** 2).sum(-1).mean()
            outputs.append(torch.stack((score, aleatoric, epistemic)))
            hidden = self.rnn(torch.cat((x, latent), -1), graph, hidden, w)
        return torch.stack(outputs)
