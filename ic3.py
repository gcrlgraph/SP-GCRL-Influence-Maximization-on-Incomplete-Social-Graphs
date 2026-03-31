"""
Information Cascade Propagation Model (IC3)
============================================
Implements a beta-activation propagation model for simulating information
spread in directed social networks.

Key components:
- Beta-activation propagation: BFS-based cascade with exposure-dependent
  activation probability. The activation probability for node v is:
    p = alpha_i * x * (1 - gamma)^(x^omega)
  where x is the exposure count, alpha_i is the edge weight, gamma is the
  global network tightness parameter, and omega controls the decay rate.
- Gamma computation: Global network structure parameter based on average
  Jaccard similarity of in-neighborhoods across all nodes.
"""

from collections import defaultdict, deque
import torch
import math
from torch_geometric.data import Data

# Cache for adjacency lists to avoid rebuilding on repeated calls
_adj_cache = {}


def compute_beta_activation(initial_Q, gamma, data, omega_i,
                            prob_scale=1.0, exponent_cap=None):
    """
    Simulate information propagation using the beta-activation model.

    The propagation follows a BFS pattern:
    1. Start from initial seed nodes (initial_Q)
    2. For each active node u, attempt to activate its out-neighbors v
    3. Activation probability depends on exposure count and edge weight:
       p(v) = sigmoid(alpha_i * x * (1 - gamma)^(x^omega) * prob_scale)
       where x = number of times v has been exposed

    Args:
        initial_Q (set or Tensor): Initial seed node set.
        gamma (float): Global network tightness parameter (from compute_gamma).
        data (Data): Graph data with edge_index, edge_weight, num_nodes.
        omega_i (Tensor): Per-node omega parameter controlling decay, shape [num_nodes].
        prob_scale (float): Scaling factor for activation probability.
        exponent_cap (float or None): Optional cap for the omega power term.

    Returns:
        tuple: (beta, active)
            - beta (Tensor): Edge activation probabilities, shape [num_edges].
            - active (Tensor): Node activation states (0/1), shape [num_nodes].
    """
    device = data.edge_index.device
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    num_nodes = data.num_nodes

    # Initialize activation states and exposure counts
    active = torch.zeros(num_nodes, dtype=torch.float, device=device)
    seen_count = torch.zeros(num_nodes, dtype=torch.int, device=device)

    # Build and cache adjacency list (CPU-side for BFS efficiency)
    cache_key = id(data)
    if cache_key not in _adj_cache:
        adj = defaultdict(list)
        ei_cpu = edge_index.cpu()
        ew_cpu = edge_weight.cpu()
        for (u, v), alpha in zip(ei_cpu.T.tolist(), ew_cpu.tolist()):
            adj[u].append((v, alpha))
        _adj_cache[cache_key] = adj
    adj_dict = _adj_cache[cache_key]

    # Initialize BFS queue with seed nodes
    current_active = deque()
    for u in initial_Q:
        active[u] = 1.0
        current_active.append(u)

    # BFS-based propagation
    while current_active:
        new_active = deque()
        for u in current_active:
            for (v, alpha_i) in adj_dict.get(u, []):
                if active[v] < 0.5:  # Node not yet activated
                    # Increment exposure count
                    seen_count[v] += 1
                    x = seen_count[v].item()

                    # Compute activation probability:
                    # p = sigmoid(alpha_i * x * (1-gamma)^(x^omega) * scale)
                    omega_power = x ** omega_i[v].item()
                    if exponent_cap is not None:
                        omega_power = min(omega_power, exponent_cap)

                    raw_prob = alpha_i * x * ((1 - gamma) ** omega_power)
                    scaled_prob = raw_prob * prob_scale
                    activation_prob = torch.sigmoid(
                        torch.tensor(scaled_prob, dtype=torch.float, device=device)
                    )

                    # Stochastic activation attempt
                    if torch.rand(1, device=device) < activation_prob:
                        active[v] = 1.0
                        new_active.append(v)

        if not new_active:
            break
        current_active = new_active

    # Compute final edge activation probabilities (beta values)
    x_values = seen_count[edge_index[0]]
    omega_powers = x_values ** omega_i[edge_index[0]]
    final_raw = edge_weight * x_values * ((1 - gamma) ** omega_powers)
    beta = torch.sigmoid(final_raw * prob_scale)

    return beta, active


def build_in_adj_list(data):
    """
    Build in-neighbor adjacency list for a directed graph.

    Args:
        data (Data): PyG data object with edge_index and num_nodes.

    Returns:
        list[set]: In-adjacency list where in_adj_list[v] = {u : (u->v) exists}.
    """
    n = data.num_nodes
    in_adj_list = [set() for _ in range(n)]

    src, dst = data.edge_index[0], data.edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        in_adj_list[v].add(u)

    return in_adj_list


def compute_gamma_directed_from_data(data):
    """
    Compute the global gamma parameter measuring network structural tightness.

    gamma is the average Jaccard similarity of in-neighborhoods:
        gamma = (1/|V'|) * sum_{j in V'} (1/|N_j|) * sum_{k in N_j} J(N_j, N_k)
    where V' = {nodes with at least one in-neighbor}, N_j = in-neighbors of j,
    and J(A, B) = |A ∩ B| / |A ∪ B| is the Jaccard similarity.

    Higher gamma indicates a more tightly connected network structure.

    Args:
        data (Data): PyG data object with graph structure.

    Returns:
        float: Global gamma value in [0, 1].
    """
    in_adj_list = build_in_adj_list(data)
    n = len(in_adj_list)
    total_value = 0.0
    count_non_empty = 0

    for j in range(n):
        N_j = in_adj_list[j]
        if len(N_j) == 0:
            continue

        sum_j = 0.0
        for k in N_j:
            N_k = in_adj_list[k]
            union_size = len(N_j.union(N_k))
            if union_size > 0:
                sum_j += len(N_j.intersection(N_k)) / union_size
            else:
                sum_j += 0.0

        avg_j = sum_j / len(N_j)
        total_value += avg_j
        count_non_empty += 1

    gamma = total_value / count_non_empty if count_non_empty > 0 else 0.0
    return gamma


