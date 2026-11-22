import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Original paper contiguous splits (For exact Planetoid, replace with ind.cora.train/val/test masks)
    # Using the standard 140/300/1000 sizes for demonstration of the splits used in the original pygcn code.
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = normalize(features)
    
    # Symmetric Normalization (Kipf & Welling)
    # A_tilde = A + I
    # D_tilde = sum(A_tilde, axis=1)
    # Output = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_symmetric(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix (Used for features)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_symmetric(mx):
    """Symmetrically normalize sparse matrix (Used for adjacency)"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    # D^{-1/2} * A * D^{-1/2}
    mx = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx

def rewire_edges(adj, rewiring_prob=1.0):
    """
    Randomly rewires edges in the graph to destroy local neighborhood structure 
    while preserving overall graph density.
    """
    print(f'Rewiring {{rewiring_prob*100}}% of edges to ablate topology...')
    adj_dense = adj.to_dense()
    num_nodes = adj_dense.shape[0]
    
    # Get all current edges (excluding self loops)
    edges = adj_dense.nonzero(as_tuple=False)
    num_edges = edges.shape[0]
    
    num_to_rewire = int(num_edges * rewiring_prob)
    
    if num_to_rewire > 0:
        # Simple uniform random rewiring for ablation test
        idx_to_remove = torch.randperm(num_edges)[:num_to_rewire]
        
        # Remove selected edges
        for idx in idx_to_remove:
            u, v = edges[idx]
            adj_dense[u, v] = 0
            adj_dense[v, u] = 0
            
        # Add random edges to maintain density
        added = 0
        while added < num_to_rewire:
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            if u != v and adj_dense[u, v] == 0:
                adj_dense[u, v] = 1
                adj_dense[v, u] = 1 # Keep symmetric
                added += 1
                
    return adj_dense.to_sparse()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
