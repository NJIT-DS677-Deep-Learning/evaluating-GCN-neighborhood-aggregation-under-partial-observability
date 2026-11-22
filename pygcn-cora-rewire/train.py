from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats

from pygcn.utils import load_data, accuracy, rewire_edges
from pygcn.models import GCN, MLP

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
# New Arguments for Ablation
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'mlp'],
                    help='Model architecture: gcn or mlp.')
parser.add_argument('--rewire', action='store_true', default=False,
                    help='Randomly rewire edges to ablate true topology.')
parser.add_argument('--seeds', type=int, default=5,
                    help='Number of random seeds to run for Welch t-test.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def train_and_test_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
    if args.rewire:
        adj = rewire_edges(adj, rewiring_prob=1.0)

    # Model selection
    ModelClass = GCN if args.model == 'gcn' else MLP
    model = ModelClass(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout)
                       
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

    def test():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test.item()

    # Train model
    for epoch in range(args.epochs):
        train(epoch)

    # Testing
    test_acc = test()
    print("Seed: {:02d}, Test Accuracy: {:.4f}".format(seed, test_acc))
    return test_acc

if __name__ == "__main__":
    print(f"\n--- Running {args.model.upper()} ({'Rewired' if args.rewire else 'True'} Topology) ---")
    test_accuracies = []
    
    base_seed = 42
    for i in range(args.seeds):
        acc = train_and_test_seed(base_seed + i)
        test_accuracies.append(acc)
        
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    
    print("\nFinal Results over {} seeds:".format(args.seeds))
    print("Mean Accuracy: {:.4f} ± {:.4f}".format(mean_acc, std_acc))
