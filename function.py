import torch
from torch_geometric.utils import degree, to_undirected

def batch_adj(X):
    # Calculate the Pearson correlation matrix
    # X: Original BOLD signal, with a shape of (batch,node,time)
    # The output is the batched Pearson correlation matrix, with a shape of (batch,node,node)
    result = []
    for x in X:
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        result.append(c)
    return  torch.stack(result, dim=0)

# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()


    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)


    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)


    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input

def construct_adjacent_matrix(pc, sparsity):
    # Construct the adjacent matrix
    # pc: The Pearson correlation matrix, with a shape of (node,node)
    # sparsity: Sparsity degree
    p = Percentile()
    thresholded = (pc > p(pc, 100 - sparsity))
    _i = thresholded.nonzero(as_tuple = False)
    _v = torch.ones(len(_i))
    _i = _i.T
    return torch.sparse.FloatTensor(_i, _v, (pc.shape[0], pc.shape[1]))

def batch_adjacent_matrix(X, sparsity, self_loop=False):
    # Construct the batched adjacent matrix
    # X: The batched Pearson correlation matrix, with a shape of (batch,node,node)
    # sparsity: Sparsity degree
    # self_loop: Whether to include self-loop when performing sparring
    result = []
    for x in X:
        if self_loop:
            a = construct_adjacent_matrix(x, sparsity)
        else:
            a = construct_adjacent_matrix(x - torch.eye(x.shape[0], x.shape[1]), sparsity).to_dense() + torch.eye(x.shape[0], x.shape[1])
        result.append(a)
    return torch.stack(result, dim=0)

def fisher_z(r):
    # Fisher z transformation
    # r: The input matrix, with a shape of (batch,node,node)
    r[r == 1] = 1 - 1e-6
    r[r == -1] = -1 + 1e-6
    return 0.5 * torch.log((1 + r) / (1 - r))

def up_triu(X, k=1):
    # Take the upper triangular matrix
    # X: The input matrix, with a shape of (batch,node,node)
    # k: The k diagonal
    # The first output is the upper triangular matrix, with a shape of (batch,node,node)
    # The second output is the mask matrix
    mask = torch.ones(X.shape[1], X.shape[2])
    mask = torch.nonzero(torch.triu(mask, k)).t()
    r = []
    for x in X:
        uptri = torch.triu(x, k)
        r.append(uptri)
    return torch.stack(r, dim=0), mask

def flatten(X, mask):
    # Flatten the matrix into a vector
    # X: The input matrix, with a shape of (batch,node,node)
    # mask: The mask matrix
    r = []
    for x in X:
        x = x[mask[0], mask[1]]
        r.append(x)
    return torch.stack(r, dim=0)


def get_edge_index(adj_mat):
    # This function builds the edge index from the adjacency matrix
    # adj_mat：the adjacent matrix, with a shape of (node,node)
    ind = torch.nonzero(adj_mat).t()
    return ind

def feature_drop_weights_dense(x, node_c):
    # Augment the node features
    # x: Original node features
    # node_c: Node degree
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s

def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    # Get the augmentation graph features
    # x: Original node features, with a shape of (node,feature)
    # w: The output of feature_drop_weights_dense
    # p: Overall probability of feature masking
    # threshold: Truncation threshold
    # The output is the augmentation graph features
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x

def degree_drop_weights(edge_index):
    # Calculate the original probability of an edge being removed
    # edge_index: With a shape of (2,_), eg. [[0,0],[0,1]]
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    # Get the edge index of the augmented graph
    # edge_index: The edge index of the original graph, with a shape of (2,_)
    # edge_weights: The original probability of an edge being removed
    # p: Overall probability of edge deletion
    # threshold: Truncation threshold
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

#######################################

def edge_index2dense(edge_index, size):
    # Restore the adjacency matrix by edge index
    # edge_index: The edge index of the graph, with a shape of (2,_)
    # size: The size of the adjacency matrix
    adj = torch.zeros(size, size)
    adj[edge_index[0], edge_index[1]] = 1.
    return adj

def add_edge_weight(W, A, n2p=True):
    # Set edge weight
    # W: The edge weight matrix
    # A: The binary adjacency matrix
    # n2p: Whether to change negative edge weight to absolute value
    if n2p: W = abs(torch.mul(A==1., W))
    else: W = torch.mul(A==1., W)
    return W

def graph_augmented(batch_adj_mat, batch_node_feature, pe, pf, threshold):
    # Execute graph augmentation
    # batch_adj_mat: The adjacency matrix of the original graph, with a shape of (batch,node,node)
    # batch_node_feature: The original graph feature matrix, with a shape of (batch,node,feature)
    # pe: Overall probability of edge deletion
    # pf: Overall probability of feature masking
    # threshold: Truncation threshold
    # The first output is the augmented adjacency matrix, with a shape of (batch,node,node)
    # The second output is the augmented node feature matrix, with a shape of (batch,node,feature)
    adj_result = []
    feature_result = []
    batch_size, adj_size = batch_adj_mat.shape[0], batch_adj_mat.shape[1]
    for i in range(batch_size):
        adj_mat = batch_adj_mat[i]
        edge_index = get_edge_index(adj_mat)
        node_feature = batch_node_feature[i]

        edge_remove_p = degree_drop_weights(edge_index)
        augmented_edge = drop_edge_weighted(edge_index=edge_index, edge_weights=edge_remove_p, p=pe, threshold=threshold)
        augmented_edge = edge_index2dense(edge_index=augmented_edge, size=adj_size)

        feature_remove_p = feature_drop_weights_dense(x=node_feature, node_c=degree(to_undirected(edge_index)[1]))
        augmented_feature = drop_feature_weighted(x=node_feature, w=feature_remove_p, p=pf, threshold=threshold)

        adj_result.append(augmented_edge)
        feature_result.append(augmented_feature)

    return torch.stack(adj_result, dim=0), torch.stack(feature_result, dim=0)
