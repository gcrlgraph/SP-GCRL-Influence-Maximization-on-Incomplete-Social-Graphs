"""
Edge Weight Calculation Module
==============================
Computes edge weights based on target node in-degree for directed graphs.
The weight of each edge (u -> v) is defined as: w(u,v) = 1 / in_degree(v),
which models the diminishing influence on nodes with many incoming connections.


"""

import torch
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def degree_calculate(data):
    """
    Compute edge weights based on target node in-degree.

    For each edge (u -> v), the weight is: w(u,v) = 1 / in_degree(v).
    A small epsilon (1e-8) is added to prevent division by zero.

    Args:
        data (torch_geometric.data.Data): Graph data with edge_index and num_nodes.

    Returns:
        torch_geometric.data.Data: New Data object with computed edge_attr (edge weights).
    """
    target_nodes = data.edge_index[1]
    in_degree = torch.bincount(target_nodes, minlength=data.num_nodes).float()

    # Edge weight = 1 / in_degree(target), with epsilon for numerical stability
    edge_weight = 1.0 / (in_degree[target_nodes] + 1e-8)

    return Data(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=edge_weight,
        num_nodes=data.num_nodes
    ).to(device)


def test_degree_calculation():
    """Unit test for edge weight calculation with a small directed graph."""
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 3, 4, 2],  # source nodes
        [1, 2, 2, 3, 2, 4, 2, 1]   # target nodes
    ], device=device)

    num_nodes = 5
    data = Data(
        x=torch.randn(num_nodes, 10),
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    weighted_data = degree_calculate(data)

    print("Target node in-degree distribution:", torch.bincount(weighted_data.edge_index[1]))
    print("Computed edge weights:", weighted_data.edge_attr)


if __name__ == "__main__":
    test_degree_calculation()
