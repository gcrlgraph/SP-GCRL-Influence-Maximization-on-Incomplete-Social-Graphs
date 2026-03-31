"""
Strategy 2: Propagation Control Based Subgraph Augmentation
============================================================
This module implements the second graph augmentation strategy for SPGCRL.

Core idea:
- Compute the propagation control score C(i) for each node using the
  transition matrix power series: W_c = sum_{t=0}^{T} P^t (P^t)^T
- C(i) = sum_j W_c(i,j), measuring how much node i can control information
  propagation across the network.
- Train a C_Predictor (inheriting from HVPredictor) to predict C(i) from
  augmented subgraphs.
- CSubgraphGenerator creates augmented views based on top-k propagation
  control nodes.
"""

import os
import random
import torch
import math
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
import itertools
from DataStrategy1 import HVPredictor
import torch_scatter
from weight_calculate import *
from data_loader import *
import numpy as np
import argparse

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_propagation_control(data, T):
    """
    Compute propagation control score C(i) for each node.

    The propagation control matrix is defined as:
        W_c = sum_{t=0}^{T} P^t @ (P^t)^T
    where P = D_out^{-1} @ A is the row-normalized transition matrix.
    C(i) = sum_j W_c(i, j) measures node i's total control over the network.

    Args:
        data (Data): Graph data with edge_index and edge_attr.
        T (int): Maximum propagation step (power series truncation).

    Returns:
        Tensor: C(i) scores for all nodes, shape [num_nodes].
    """
    data = data.to(device)
    num_nodes = data.num_nodes

    edge_index, edge_attr = data.edge_index, data.edge_attr
    # Build dense adjacency matrix
    A = torch.sparse_coo_tensor(edge_index, edge_attr, (num_nodes, num_nodes)).to(device).to_dense()

    # Row-normalize: P = D_out^{-1} @ A
    out_degree = A.sum(dim=1)
    D_inv = torch.diag(1.0 / (out_degree + 1e-8))
    P = D_inv @ A

    # Compute W_c = sum_{t=0}^{T} P^t @ (P^t)^T
    Wc = torch.zeros_like(P, device=device)
    for t in range(T + 1):
        Pt = torch.matrix_power(P, t) if t > 0 else torch.eye(num_nodes, device=device)
        Wc += Pt @ Pt.T

    # C(i) = row sum of W_c
    return Wc.sum(dim=1)


class C_Predictor(HVPredictor):
    """
    Predictor for propagation control score C(i).
    Inherits the attention-based message passing from HVPredictor.

    Args:
        feature_dim (int): Dimension of each feature half (S or T).
        hidden_dim (int): Hidden dimension for attention projection.
    """

    def __init__(self, feature_dim, hidden_dim):
        super().__init__(feature_dim, hidden_dim)
        self.regress = nn.Linear(2 * feature_dim, 1)


class CSubgraphGenerator:
    """
    Generates augmented subgraph views based on propagation control C(i).

    The augmentation process:
    1. Select top-k nodes by C(i) as critical nodes
    2. Keep all edges connected to critical nodes
    3. Randomly sample a fraction of remaining edges
    4. Apply random feature masking

    Args:
        predictor (C_Predictor): The predictor model.
        feature_mask_ratio (float): Fraction of features to mask.
        edge_keep_ratio (float): Fraction of non-critical edges to keep.
        k_predictor (int): Number of message passing iterations for the predictor.
    """

    def __init__(self, predictor, feature_mask_ratio, edge_keep_ratio, k_predictor):
        self.predictor = predictor
        self.feature_mask_ratio = feature_mask_ratio
        self.edge_keep_ratio = edge_keep_ratio
        self.k_predictor = k_predictor

    def generate_subgraph(self, data, C, k):
        """
        Generate an augmented subgraph based on propagation control scores.

        Args:
            data (Data): Original graph data.
            C (Tensor): Propagation control scores for all nodes.
            k (int): Number of top-C(i) nodes to select as critical (k_subgraph).

        Returns:
            Data: Augmented subgraph data.
        """
        data = data.to(device)
        _, topk_indices = torch.topk(C, k)

        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Identify critical edges (connected to top-k nodes)
        src_mask = torch.isin(edge_index[0], topk_indices)
        dst_mask = torch.isin(edge_index[1], topk_indices)
        critical_edges = src_mask | dst_mask

        # Randomly sample non-critical edges
        non_critical = torch.where(~critical_edges)[0]
        num_keep = int(len(non_critical) * self.edge_keep_ratio)
        kept_non_critical = non_critical[torch.randperm(len(non_critical), device=device)[:num_keep]]

        final_edges = torch.cat([
            torch.where(critical_edges)[0],
            kept_non_critical
        ])

        # Apply random feature masking
        x_masked = data.x.clone()
        s_dim = x_masked.shape[1] // 2

        s_mask = torch.rand_like(x_masked[:, :s_dim], device=device) < self.feature_mask_ratio
        x_masked[:, :s_dim][s_mask] = 0

        t_mask = torch.rand_like(x_masked[:, s_dim:]) < self.feature_mask_ratio
        x_masked[:, s_dim:][t_mask] = 0

        return Data(
            x=x_masked,
            edge_index=edge_index[:, final_edges],
            edge_attr=edge_attr[final_edges],
            num_nodes=data.num_nodes
        ).to(device)


# ============================================================================
# Training Loop
# ============================================================================

def train_c_predictor(
        data, hidden_dim, num_epochs, lr,
        k_subgraph, k_predictor, T_control,
        feature_mask_ratio, edge_keep_ratio,
        save_model_path, print_every, patience=100
):
    """
    Train the C_Predictor model with early stopping.

    Args:
        data (Data): Graph data with features and edge weights.
        hidden_dim (int): Hidden dimension for the predictor.
        num_epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        k_subgraph (int): Number of top-C(i) nodes for subgraph generation.
        k_predictor (int): Number of message passing iterations.
        T_control (int): Maximum propagation step for computing C(i).
        feature_mask_ratio (float): Feature masking ratio.
        edge_keep_ratio (float): Edge retention ratio for non-critical edges.
        save_model_path (str): Path to save the best model checkpoint.
        print_every (int): Print frequency.
        patience (int): Early stopping patience.

    Returns:
        C_Predictor: The trained predictor model.
    """
    actual_feature_dim = data.x.size(1) // 2
    predictor = C_Predictor(feature_dim=actual_feature_dim, hidden_dim=hidden_dim).to(device)
    generator = CSubgraphGenerator(
        predictor,
        feature_mask_ratio=feature_mask_ratio,
        edge_keep_ratio=edge_keep_ratio,
        k_predictor=k_predictor
    )
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    best_loss = float('inf')
    data = data.to(device)
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        C = compute_propagation_control(data, T=T_control)
        sub_data = generator.generate_subgraph(data, C, k=k_subgraph)

        pred = predictor(sub_data, k=k_predictor).squeeze()
        loss = torch.nn.functional.mse_loss(pred, C)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(predictor.state_dict(), save_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
            break

        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:03d}/{num_epochs-1} | Loss: {current_loss:.4f} | Best: {best_loss:.4f}")

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Best model saved to: {save_model_path}")
    return predictor


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C Predictor (Strategy 2)")

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='fb_politician',
                        help='Dataset identifier (see data_loader.py for options)')
    parser.add_argument('--feature_dim', type=int, default=32,
                        help='Node feature dimension (each half of S/T)')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Max training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--k_subgraph', type=int, default=10,
                        help='Number of top-C(i) nodes for subgraph generation')
    parser.add_argument('--k_predictor', type=int, default=5,
                        help='Number of message passing iterations')
    parser.add_argument('--T_control', type=int, default=5,
                        help='Max propagation step for C(i) computation')
    parser.add_argument('--feature_mask_ratio', type=float, default=0.2,
                        help='Fraction of features to mask')
    parser.add_argument('--edge_keep_ratio', type=float, default=0.8,
                        help='Fraction of non-critical edges to keep')
    parser.add_argument('--print_every', type=int, default=20, help='Print frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    print(f"\n=== Loading dataset: {args.dataset_name} ===")
    data, _ = load_data(args.dataset_name)
    data.x = torch.randn((data.num_nodes, args.feature_dim * 2), device=device)
    data = degree_calculate(data)

    # Save path (relative to project root)
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', 'C_save')
    save_model_path = os.path.join(save_dir, f"{args.dataset_name}_c_predictor_best.pth")

    # Train
    train_c_predictor(
        data=data,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        lr=args.lr,
        k_subgraph=args.k_subgraph,
        k_predictor=args.k_predictor,
        T_control=args.T_control,
        feature_mask_ratio=args.feature_mask_ratio,
        edge_keep_ratio=args.edge_keep_ratio,
        save_model_path=save_model_path,
        print_every=args.print_every,
        patience=args.patience,
    )
