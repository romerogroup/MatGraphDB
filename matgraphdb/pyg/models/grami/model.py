import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
from torch_geometric.nn import HeteroConv, GATConv
from torch.nn.parameter import Parameter
import warnings
# Make sure to update this function to work with PyG’s HeteroData
from matgraphdb.pyg.models.grami.utils import feature_in_graph  
torch.set_printoptions(profile="full")
warnings.filterwarnings("ignore")



def weight_variable_glorot(input_dim, output_dim):
    """Create a weight variable using Glorot initialization."""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.FloatTensor(input_dim, output_dim).uniform_(-init_range, init_range)
    return nn.Parameter(initial, requires_grad=True)

class Dense(nn.Module):
    """A simple dense (fully-connected) layer."""
    def __init__(self, input_dim, output_dim, dropout=0., bias=True, act=F.relu):
        super(Dense, self).__init__()
        self.dropout = dropout
        self.act = act
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, inputs):
        x = F.dropout(inputs, self.dropout, training=self.training)
        x = self.linear(x)
        x = self.act(x)
        return x

class RGATLayer(nn.Module):
    """
    A relational (i.e. heterogeneous) graph attention layer built on PyG.
    For each relation (edge type) we create a GATConv, and then wrap them with a HeteroConv.
    """
    def __init__(self, in_feats, out_feats, num_heads, rel_names, act=lambda x: x):
        super(RGATLayer, self).__init__()
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.act = act
        convs = {}
        for rel in rel_names:
            convs[rel] = GATConv(in_feats, out_feats // num_heads, heads=num_heads, dropout=0.2)
        self.convs = nn.ModuleDict(convs)
        self.hetero_conv = HeteroConv({rel: self.convs[rel] for rel in rel_names}, aggr='mean')

    def forward(self, dataset, data, inputs):
        """
        dataset: string identifier (e.g. 'ACM', etc.) used to fix an ordering of node types
        data: a torch_geometric.data.HeteroData object (provides edge_index_dict, node_types, etc.)
        inputs: a list of dictionaries mapping node types to feature tensors
        """
        outputs = []
        if dataset == 'ACM':
            key_order = ['paper', 'author', 'subject']
        elif dataset == 'DBLP':
            key_order = ['author', 'paper', 'term', 'venue']
        elif dataset == 'YELP':
            key_order = ['business', 'user', 'service', 'level']
        else:
            key_order = list(data.node_types)
        for inp in inputs:
            h = self.hetero_conv(inp, data.edge_index_dict)
            out_list = []
            for k in key_order:
                if k in h:
                    out = self.act(h[k])
                    # Reshape to [num_nodes, out_feats]
                    out = out.view(-1, self.out_feats)
                    out_list.append(out)
            out_cat = torch.cat(out_list, dim=0)
            outputs.append(out_cat)
        outputs = torch.stack(outputs, dim=0)
        return outputs




def MinMaxScalar(x):
    min_vals, _ = torch.min(x, dim=1, keepdim=True)
    max_vals, _ = torch.max(x, dim=1, keepdim=True)
    scaled_x = (x - min_vals) / (max_vals - min_vals)
    return scaled_x


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class GraMI(nn.Module):
    def __init__(self, dataset, data, src_node, feats_dim_list, num_nodes, ndim,
                 input_feat_dim, hidden_dim, hidden_dim1, hidden_dim2,
                 num_heads, dropout, encsto='semi', gdc='ip',
                 ndist='Bernoulli', copyK=1, copyJ=1, device='cuda'):
        """
        dataset: string identifier (e.g. 'ACM', 'DBLP', etc.)
        data: a torch_geometric.data.HeteroData object
        src_node: the index of the “source” node type for your application
        feats_dim_list: list of input dimensions for each modality
        num_nodes: total number of nodes (or other appropriate measure)
        ndim: latent noise dimension
        input_feat_dim: original feature dimension (for recovery)
        hidden_dim, hidden_dim1, hidden_dim2: hidden layer sizes
        num_heads: number of attention heads
        dropout: dropout probability
        encsto, gdc, ndist, copyK, copyJ, device: other options
        """
        super(GraMI, self).__init__()
        self.dataset = dataset
        # Create one fully connected layer per modality
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else lambda x: x
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.ac = base_HGNN(dataset, data, num_nodes, ndim, input_feat_dim, hidden_dim,
                             hidden_dim1, hidden_dim2, num_heads, dropout, src_node,
                             encsto, gdc, ndist, copyK, copyJ, device)

    def forward(self, features_list, data):
        """
        features_list: a list of feature tensors (one per modality or node type)
        data: a torch_geometric.data.HeteroData object
        """
        x_all = []
        for i in range(len(features_list)):
            # Apply a linear layer + tanh nonlinearity with dropout
            x = torch.tanh(self.feat_drop(self.fc_list[i](features_list[i])))
            # Add a “batch” (or modality) dimension so that later we can combine them
            x_all.append(x.unsqueeze(0))
        # Concatenate the modalities along a new dimension
        x = torch.cat(x_all, dim=1)  # shape: [1, num_modalities, hidden_dim]
        label_a = x.squeeze(0)
        # Assume that feature_in_graph now converts your PyG HeteroData object using the new features.
        # (You must adjust feature_in_graph to work with PyG.)
        features_dict = feature_in_graph(self.dataset, data, x_all, device=data.x.device)
        ac = self.ac(x, data, features_dict)
        return label_a, ac


class base_HGNN(nn.Module):
    def __init__(self, dataset, data, num_nodes, ndim, input_feat_dim, hidden_dim,
                 hidden_dim1, hidden_dim2, num_heads, dropout, src_node,
                 encsto='semi', gdc='ip', ndist='Bernoulli', copyK=1, copyJ=1,
                 device='cuda'):
        super(base_HGNN, self).__init__()
        self.dataset = dataset
        self.n_samples = num_nodes
        self.data = data  # a torch_geometric.data.HeteroData object
        # In PyG, you typically have a dict of edge_index per relation.
        self.rel_names = list(data.edge_index_dict.keys())
        ndim = hidden_dim  # reset for later layers
        self.gat_e = RGATLayer(in_feats=ndim, out_feats=hidden_dim1, num_heads=num_heads,
                               rel_names=self.rel_names, act=F.relu)
        self.gat_1 = RGATLayer(in_feats=hidden_dim, out_feats=hidden_dim1, num_heads=num_heads,
                               rel_names=self.rel_names, act=F.relu)
        self.gat_2 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=num_heads,
                               rel_names=self.rel_names, act=lambda x: x)
        self.gat_3 = RGATLayer(in_feats=hidden_dim1, out_feats=hidden_dim2, num_heads=num_heads,
                               rel_names=self.rel_names, act=lambda x: x)
        # Feature embedding MLPs
        self.mlpe = Dense(input_dim=ndim, output_dim=hidden_dim1, dropout=dropout, act=torch.tanh)
        self.mlp1 = Dense(input_dim=self.n_samples, output_dim=hidden_dim1, act=torch.tanh)
        self.mlp2 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)
        self.mlp3 = Dense(input_dim=hidden_dim1, output_dim=hidden_dim2, dropout=dropout, act=lambda x: x)
        self.mlp_recover = Dense(input_dim=hidden_dim, output_dim=input_feat_dim, dropout=dropout,
                                 act=torch.sigmoid)
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.dc2 = GraphDecoder2(hidden_dim2, dropout, gdc=gdc)
        self.device = device

        if ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':
            self.ndist = tdist.Normal(torch.tensor([0.], device=self.device),
                                      torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        self.src_node = src_node
        self.reweight = ((self.ndim + hidden_dim1) / (hidden_dim + hidden_dim1)) ** (.5)

    def node_encode(self, data, x_all):
        """
        x_all: here we assume a dict {node_type: feature_tensor} (for example,
               returned by your new feature_in_graph function)
        """
        # Make a “noisy” copy of the graph (data_noise should be a HeteroData)
        data_noise = data.clone()
        data_noise = feature_in_graph(self.dataset, data_noise, x_all, device=self.device)
        # Build an input dictionary from the node features (each node type’s features are stored in .x)
        inputs = {ntype: data_noise[ntype].x for ntype in data_noise.node_types}
        # Run one heterogeneous attention layer (note: here we pass a list with one dict)
        hiddenx = self.gat_1(self.dataset, data_noise, [inputs])
        
        # Create “noise” for each node type (this example uses a simple mean over samples;
        # you may wish to modify this so that you keep multiple copies, etc.)
        noise_list = {}
        if self.ndim >= 1:
            for ntype, feat in x_all.items():
                e = self.ndist.sample(torch.Size([self.K + self.J, feat.size(0), self.ndim])).squeeze(-1)
                e = e * self.reweight
                # Here we simply average over the K+J copies (adjust as needed)
                noise_list[ntype] = e.mean(dim=0)
            noise_dict = noise_list
            hiddene = self.gat_e(self.dataset, data_noise, [noise_dict])
        else:
            print("no randomness.")
            hiddene = {ntype: torch.zeros_like(x_all[ntype]) for ntype in x_all.keys()}
        # Combine the two sets of hidden features (again, adjust the details as needed)
        hidden1 = {}
        for ntype in x_all.keys():
            hidden1[ntype] = hiddenx[0][ntype] + hiddene[ntype]
        # For the “stochastic” encoding, choose either the full or partial hidden representation
        hidden_std1 = {ntype: (hidden1[ntype] if self.encsto == 'full' else hiddenx[0][ntype])
                       for ntype in x_all.keys()}
        mu = self.gat_2(self.dataset, data_noise, [hidden1])
        logvar = self.gat_3(self.dataset, data_noise, [hidden_std1])

        # Now, split (or reassemble) the tensors per node type.
        # For example, for ACM assume the order is: 'paper', 'author', 'subject'
        if self.dataset == 'ACM':
            num_paper = data['paper'].num_nodes
            num_author = data['author'].num_nodes
            num_subject = data['subject'].num_nodes
            # Concatenate the outputs and then split by the known node counts
            mu_cat = torch.cat([mu[0]['paper'], mu[0]['author'], mu[0]['subject']], dim=0)
            logvar_cat = torch.cat([logvar[0]['paper'], logvar[0]['author'], logvar[0]['subject']], dim=0)
            mu = torch.split(mu_cat, [num_paper, num_author, num_subject], dim=0)
            logvar = torch.split(logvar_cat, [num_paper, num_author, num_subject], dim=0)
        # (Similar adjustments would be needed for DBLP and YELP.)
        return mu, logvar

    def feature_encode(self, x):
        """
        x: assumed to be of shape [1, num_modalities, feature_dim]
        """
        # Transpose and squeeze to get a 2D tensor (you may need to adjust the dimensions)
        f = x.transpose(1, 2).squeeze(0)
        hiddenf = self.mlp1(f)
        if self.ndim >= 1:
            e = self.ndist.sample(torch.Size([self.K + self.J, f.size(0), self.ndim])).squeeze(-1)
            e = e * self.reweight
            hiddene = self.mlpe(e.mean(dim=0))
        else:
            print("no randomness.")
            hiddene = torch.zeros_like(hiddenf)
        hidden1 = hiddenf + hiddene
        muf = self.mlp2(hidden1)
        hidden_sd = hidden1 if self.encsto == 'full' else hiddenf
        logvarf = self.mlp3(hidden_sd)
        return muf, logvarf

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu), eps

    def forward(self, x, data, x_all):
        """
        x: the stacked feature tensor from GraMI.forward
        data: the heterogeneous graph (HeteroData)
        x_all: a dictionary {node_type: feature_tensor}
        """
        src_node = self.src_node
        mu_list, logvar_list = self.node_encode(data, x_all)
        muf, logvarf = self.feature_encode(x)
        # (For simplicity the following “slicing” assumes that the first K rows are to be dropped.)
        emb_mu_list = [mu[self.K:] for mu in mu_list]
        emb_logvar_list = [logvar[self.K:] for logvar in logvar_list]
        emb_muf = muf[self.K:]
        emb_logvarf = logvarf[self.K:]
        z_all, eps_all = [], []
        for emb_mu, emb_logvar in zip(emb_mu_list, emb_logvar_list):
            z, eps = self.reparameterize(emb_mu, emb_logvar)
            z_all.append(z)
            eps_all.append(eps)
        zf, epsf = self.reparameterize(emb_muf, emb_logvarf)
        pred_adj_all, z_scaled1_all, z_scaled2_all, rk_all = [], [], [], []
        for i in range(len(z_all)):
            if i == src_node:
                continue
            else:
                adj_, z_scaled1, z_scaled2, rk = self.dc(z_all[src_node], z_all[i])
                pred_adj_all.append(adj_)
                z_scaled1_all.append(z_scaled1)
                z_scaled2_all.append(z_scaled2)
                rk_all.append(rk)
        Za = torch.cat(z_all, dim=1)
        pred_a, z_scaleda, z_scaledf, _ = self.dc2(Za, zf)
        if src_node == 0:
            fea_recover = pred_a[:, :mu_list[src_node].size(0), :]
        else:
            prev_nodes = mu_list[src_node - 1].size(0)
            curr_nodes = mu_list[src_node].size(0)
            fea_recover = pred_a[:, prev_nodes:prev_nodes + curr_nodes, :]
        fea_recover = self.mlp_recover(fea_recover)
        return (pred_adj_all, pred_a, fea_recover, mu_list, muf, logvar_list,
                logvarf, z_all, zf, Za, z_scaled1_all, z_scaled2_all, z_scaledf,
                z_scaleda, eps_all, epsf, rk_all)


class GraphDecoder(nn.Module):
    """Decoder that uses (optionally dropped out) inner products between latent representations."""
    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(1, zdim))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z1, z2):
        z1 = F.dropout(z1, self.dropout, training=self.training)
        z2 = F.dropout(z2, self.dropout, training=self.training)
        rk = torch.sigmoid(self.rk_lgt).pow(.5)
        adj_lgt = torch.bmm(z1, z2.transpose(1, 2))
        if self.gdc == 'ip':
            adj = torch.sigmoid(adj_lgt)
        elif self.gdc == 'bp':
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())
        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)
        return adj, z1, z2, rk.pow(2)


class GraphDecoder2(nn.Module):
    """Another decoder variant for feature reconstruction."""
    def __init__(self, zdim, dropout, gdc='ip'):
        super(GraphDecoder2, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = Parameter(torch.FloatTensor(1, zdim))
        self.reset_parameters()
        self.SMALL = 1e-16

    def reset_parameters(self):
        nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z1, z2):
        z1 = F.dropout(z1, self.dropout, training=self.training)
        z2 = F.dropout(z2, self.dropout, training=self.training)
        rk = torch.sigmoid(self.rk_lgt).pow(.5)
        adj_lgt = torch.bmm(z1, z2.transpose(1, 2))
        if self.gdc == 'ip':
            adj = torch.tanh(adj_lgt)
        elif self.gdc == 'bp':
            adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
            adj = 1 - torch.exp(-adj_lgt.exp())
        if not self.training:
            adj = torch.mean(adj, dim=0, keepdim=True)
        return adj, z1, z2, rk.pow(2)
