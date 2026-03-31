"""
Strategy 1: Path Inverse Entropy (HV) Based Subgraph Augmentation
==================================================================
This module implements the first graph augmentation strategy for SPGCRL.

Core idea:
- Compute the path inverse entropy H(v) for each node, which measures the
  diversity of shortest-path weights from predecessors to node v.
- Train an HVPredictor neural network (with attention-like message passing)
  to predict H(v) from augmented subgraphs.
- The SubgraphGenerator creates augmented graph views by:
  1. Selecting top-k nodes by H(v) as critical nodes
  2. Preserving all edges connected to critical nodes
  3. Randomly sampling a fraction of remaining edges
  4. Applying random feature masking
"""

import math
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
import itertools
import torch_scatter
from weight_calculate import *
from data_loader import *
import matplotlib.pyplot as plt
import numpy as np
import random
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


class HVPredictor(nn.Module):
    """
    Neural network that predicts path inverse entropy H(v) for each node.
    Uses attention-like message passing with learnable aggregation weights.

    Node features are split into S (source) and T (target) halves,
    aggregated along incoming and outgoing edges respectively.

    Args:
        feature_dim (int): Dimension of each feature half (S or T).
        hidden_dim (int): Hidden dimension for attention projection.
    """

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        # Shared linear projection for attention computation
        self.W = nn.Linear(feature_dim, hidden_dim, bias=False)

        # Attention parameters for source->target and target->source directions
        self.alpha = nn.Parameter(torch.Tensor(2 * hidden_dim))   # S->T attention
        self.epsilon = nn.Parameter(torch.Tensor(2 * hidden_dim)) # T->S attention

        # Learnable aggregation weights for combining edge weights and attention
        self.lambda1 = nn.Parameter(torch.Tensor(1))  # edge weight coeff (outgoing)
        self.lambda2 = nn.Parameter(torch.Tensor(1))  # attention coeff (outgoing)
        self.rho1 = nn.Parameter(torch.Tensor(1))     # edge weight coeff (incoming)
        self.rho2 = nn.Parameter(torch.Tensor(1))     # attention coeff (incoming)

        # Final regression head: maps concatenated [S, T] to scalar H(v)
        self.regress = nn.Linear(2 * feature_dim, 1)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        """Initialize all learnable parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.alpha)
        nn.init.normal_(self.epsilon)
        nn.init.constant_(self.lambda1, 0.5)
        nn.init.constant_(self.lambda2, 0.5)
        nn.init.constant_(self.rho1, 0.5)
        nn.init.constant_(self.rho2, 0.5)

    def forward(self, data, k):
        """
        Forward pass with k rounds of message passing.

        Args:
            data (Data): Graph data with x (node features), edge_index, edge_attr.
            k (int): Number of message passing iterations.

        Returns:
            Tensor: Predicted H(v) for each node, shape [num_nodes, 1].
        """
        edge_index = data.edge_index
        edge_weight = data.edge_attr
        # Split features into source (S) and target (T) halves
        S = data.x[:, :data.x.size(1) // 2]
        T = data.x[:, data.x.size(1) // 2:]

        for _ in range(k):
            src, dst = edge_index

            # --- S->T attention (phi): how much source S influences target T ---
            S_proj = self.W(S[src])
            T_proj = self.W(T[dst])
            d_uv = (torch.cat([S_proj, T_proj], dim=1) * self.alpha).sum(dim=1)
            phi = torch_scatter.scatter(
                torch.exp(torch.nn.functional.leaky_relu(d_uv)),
                dst, dim=0, dim_size=data.num_nodes
            )
            phi = torch.exp(d_uv) / (phi[dst] + 1e-8)

            # --- T->S attention (alpha): how much target T influences source S ---
            T_proj_src = self.W(T[src])
            S_proj_dst = self.W(S[dst])
            f_vw = (torch.cat([T_proj_src, S_proj_dst], dim=1) * self.epsilon).sum(dim=1)
            alpha = torch_scatter.scatter(
                torch.exp(torch.nn.functional.leaky_relu(f_vw)),
                src, dim=0, dim_size=data.num_nodes
            )
            alpha = torch.exp(f_vw) / (alpha[src] + 1e-8)

            # --- Aggregate: combine edge weights and attention scores ---
            b = torch_scatter.scatter(
                self.lambda1 * edge_weight + self.lambda2 * alpha,
                src, dim=0, dim_size=data.num_nodes
            )
            c = torch_scatter.scatter(
                self.rho1 * edge_weight + self.rho2 * phi,
                dst, dim=0, dim_size=data.num_nodes
            )

            # Update S and T with gated aggregation
            S = torch.sigmoid(b.unsqueeze(1) * S)
            T = torch.sigmoid(c.unsqueeze(1) * T)

        return self.regress(torch.cat([S, T], dim=1))


class SubgraphGenerator:
    """
    Generates augmented subgraph views based on path inverse entropy H(v).

    The augmentation process:
    1. Compute H(v) for all nodes
    2. Select top-k nodes by H(v) as critical nodes
    3. Keep all edges connected to critical nodes
    4. Randomly sample a fraction of remaining edges
    5. Apply random feature masking to both S and T feature halves

    Args:
        predictor (HVPredictor): The predictor model (used for loss computation).
        feature_mask_ratio (float): Fraction of features to mask (set to 0).
        edge_keep_ratio (float): Fraction of non-critical edges to keep.
        k_predictor (int): Number of message passing iterations for the predictor.
    """

    def __init__(self, predictor, feature_mask_ratio, edge_keep_ratio, k_predictor):
        self.predictor = predictor
        self.loss_fn = nn.MSELoss()
        self.feature_mask_ratio = feature_mask_ratio
        self.edge_keep_ratio = edge_keep_ratio
        self.k_predictor = k_predictor

    def compute_hv(self, data):
        """
        Compute path inverse entropy H(v) for each node.

        H(v) = -sum_{u in pred(v)} p(u,v) * log(p(u,v))
        where p(u,v) = (1/d(u,v)) / sum_{w in pred(v)} (1/d(w,v))
        and d(u,v) = -log(edge_weight(u,v)) is the shortest-path distance.

        Args:
            data (Data): Graph data with edge_index and edge_attr.

        Returns:
            Tensor: H(v) values for all nodes, shape [num_nodes].
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(data.num_nodes))
        edge_weight = data.edge_attr.clamp(min=1e-5)
        d_uv = -torch.log(edge_weight).cpu().numpy()

        for (u, v), d in zip(data.edge_index.T.cpu().numpy(), d_uv):
            G.add_edge(u, v, weight=max(d, 1e-8))

        H = torch.zeros(data.num_nodes, device=device)
        for v in range(data.num_nodes):
            predecessors = list(G.predecessors(v))
            if not predecessors:
                continue
            inv_d = [1.0 / G[u][v]['weight'] for u in predecessors]
            Z_v = sum(inv_d)
            p_uv = [1.0 / (Z_v * G[u][v]['weight']) for u in predecessors]
            p_tensor = torch.tensor(p_uv, dtype=torch.float32, device=device)
            H[v] = -torch.sum(p_tensor * torch.log(p_tensor))

        return H

    def generate_subgraph(self, data, k):
        """
        Generate an augmented subgraph view and compute prediction loss.

        Args:
            data (Data): Original graph data.
            k (int): Number of top-H(v) nodes to select as critical (k_subgraph).

        Returns:
            tuple: (augmented_data, loss) where loss is MSE between predicted and true H(v).
        """
        data = data.to(device)
        H = self.compute_hv(data)
        _, topk_indices = torch.topk(H, k)

        edge_index = data.edge_index
        edge_weight = data.edge_attr

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

        t_mask = torch.rand_like(x_masked[:, s_dim:], device=device) < self.feature_mask_ratio
        x_masked[:, s_dim:][t_mask] = 0

        sub_data = Data(
            x=x_masked,
            edge_index=edge_index[:, final_edges],
            edge_attr=edge_weight[final_edges],
            num_nodes=data.num_nodes
        ).to(device)

        # Compute prediction loss
        pred_hv = self.predictor(sub_data, k=self.k_predictor)
        loss = self.loss_fn(pred_hv.squeeze(), H)

        return sub_data, loss


# ============================================================================
# Training Loop
# ============================================================================

def train_predictor(
        data, hidden_dim, num_epochs, lr,
        k_subgraph, k_predictor,
        feature_mask_ratio, edge_keep_ratio,
        save_model_path, print_every, patience=100
):
    """
    Train the HVPredictor model with early stopping.

    Args:
        data (Data): Graph data with features and edge weights.
        hidden_dim (int): Hidden dimension for the predictor.
        num_epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        k_subgraph (int): Number of top-H(v) nodes for subgraph generation.
        k_predictor (int): Number of message passing iterations.
        feature_mask_ratio (float): Feature masking ratio for augmentation.
        edge_keep_ratio (float): Edge retention ratio for non-critical edges.
        save_model_path (str): Path to save the best model checkpoint.
        print_every (int): Print training progress every N epochs.
        patience (int): Early stopping patience (epochs without improvement).

    Returns:
        HVPredictor: The trained predictor model.
    """
    actual_feature_dim = data.x.size(1) // 2
    predictor = HVPredictor(feature_dim=actual_feature_dim, hidden_dim=hidden_dim).to(device)
    generator = SubgraphGenerator(
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
        sub_data, loss = generator.generate_subgraph(data, k=k_subgraph)
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
    parser = argparse.ArgumentParser(description="Train HV Predictor (Strategy 1)")

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
                        help='Number of top-H(v) nodes for subgraph generation')
    parser.add_argument('--k_predictor', type=int, default=5,
                        help='Number of message passing iterations')
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
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', 'HV_save')
    save_model_path = os.path.join(save_dir, f"{args.dataset_name}_hv_predictor_best.pth")

    # Train
    train_predictor(
        data=data,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        lr=args.lr,
        k_subgraph=args.k_subgraph,
        k_predictor=args.k_predictor,
        feature_mask_ratio=args.feature_mask_ratio,
        edge_keep_ratio=args.edge_keep_ratio,
        save_model_path=save_model_path,
        print_every=args.print_every,
        patience=args.patience,
    )
