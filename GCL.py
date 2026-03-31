import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from DataStrategy1 import SubgraphGenerator as Strategy1Generator
from DataStrategy2 import CSubgraphGenerator as Strategy2Generator
from DataStrategy1 import HVPredictor
from DataStrategy2 import C_Predictor, compute_propagation_control
from data_loader import *
from weight_calculate import *  
import argparse
import time
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
class GCLModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim, tau=0.2):
        super().__init__()
        self.gnn = GNNEncoder(2 * feat_dim, hidden_dim, out_dim)  
        self.proj = nn.Sequential(
            nn.Linear(out_dim, 2*out_dim),
            nn.ReLU(),
            nn.Linear(2*out_dim, out_dim)
        )
        self.tau = tau
        self.q_net = nn.Linear(out_dim, 1)

    def get_embeddings(self, g1, g2):
        return self.forward(g1, g2)
    
    def predict_q_values(self, embeddings):
        return self.q_net(embeddings)

    def forward(self, g1, g2):
        g1.x, g1.edge_index, g1.edge_weight = self._ensure_device(g1)
        g2.x, g2.edge_index, g2.edge_weight = self._ensure_device(g2)
        h1 = self.gnn(g1.x, g1.edge_index, g1.edge_weight)
        h2 = self.gnn(g2.x, g2.edge_index, g2.edge_weight)
        return h1, h2
    def _ensure_device(self, graph):
        return (graph.x.to(device), 
                graph.edge_index.to(device),
                graph.edge_weight.to(device) if graph.edge_weight is not None else None)


    def contrastive_loss(self, u, v):
        u_proj = F.normalize(self.proj(u), dim=-1)  # shape [N, d]
        v_proj = F.normalize(self.proj(v), dim=-1)  # shape [N, d]
        batch_size = u_proj.size(0)

        # Compute all positive pair similarities
        pos_sim = torch.sum(u_proj * v_proj, dim=-1)  # [N]
        pos_exp = torch.exp(pos_sim / self.tau)  # numerator

        # All pairwise similarity matrices
        sim_uv = torch.mm(u_proj, v_proj.T) / self.tau  # [N, N]
        sim_uu = torch.mm(u_proj, u_proj.T) / self.tau  # [N, N]

        # Mask out self-similarity
        eye = torch.eye(batch_size, dtype=torch.bool, device=u.device)
        sim_uv = sim_uv.masked_fill(eye, -float('inf'))
        sim_uu = sim_uu.masked_fill(eye, -float('inf'))

        # Denominator (positive + all negatives)
        denom = pos_exp + sim_uv.exp().sum(dim=1) + sim_uu.exp().sum(dim=1)

        # Compute final loss
        loss_u = -torch.log(pos_exp / denom)

        # Symmetric term loss(v, u)
        sim_vu = torch.mm(v_proj, u_proj.T) / self.tau
        sim_vv = torch.mm(v_proj, v_proj.T) / self.tau
        sim_vu = sim_vu.masked_fill(eye, -float('inf'))
        sim_vv = sim_vv.masked_fill(eye, -float('inf'))

        pos_sim_rev = torch.sum(v_proj * u_proj, dim=-1)
        pos_exp_rev = torch.exp(pos_sim_rev / self.tau)
        denom_rev = pos_exp_rev + sim_vu.exp().sum(dim=1) + sim_vv.exp().sum(dim=1)
        loss_v = -torch.log(pos_exp_rev / denom_rev)

        # Symmetric average
        return (loss_u.mean() + loss_v.mean()) / 2 

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.out_dim = out_dim 
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(x, edge_index, edge_weight)

def train_integrated(args):
    # Load dataset
    print(f"\n=== Loading dataset: {args.dataset_name} ===")
    data, _ = load_data(args.dataset_name)
    num_nodes = data.num_nodes
    
    # Add random features to dataset
    data.x = torch.randn((num_nodes, 2 * args.num_features), device=device)
    
    # Compute edge weights
    data = degree_calculate(data)
    print(f"Dataset loaded. Nodes: {num_nodes}, Edges: {data.edge_index.size(1)}")
    
    # Pretrained model weight paths
    hv_weights_path = args.hv_weights_path
    c_weights_path = args.c_weights_path
    gcl_save_path = args.gcl_save_path if args.gcl_save_path else f"gcl_models/{args.dataset_name}_best_gcl_model.pth"
    import os
    os.makedirs(os.path.dirname(gcl_save_path), exist_ok=True)

    print("\n=== Loading pretrained models ===")
    print(f"HV Predictor weights: {hv_weights_path}")
    print(f"C Predictor weights: {c_weights_path}")
    print(f"GCL model save path: {gcl_save_path}")
    
    # Load pretrained HV Predictor
    try:
        hv_predictor = HVPredictor(feature_dim=args.num_features, hidden_dim=args.hidden_dim).to(device)
        hv_predictor.load_state_dict(torch.load(hv_weights_path, weights_only=True))
        print("Successfully loaded HV Predictor weights")
        print(f"HV Predictor parameters: {sum(p.numel() for p in hv_predictor.parameters())}")
    except Exception as e:
        print(f"Error loading HV Predictor weights: {e}")
        sys.exit(1)

    # Load pretrained C Predictor
    try:
        c_predictor = C_Predictor(feature_dim=args.num_features, hidden_dim=args.hidden_dim).to(device)
        c_predictor.load_state_dict(torch.load(c_weights_path, weights_only=True))
        print("Successfully loaded C Predictor weights")
        print(f"C Predictor parameters: {sum(p.numel() for p in c_predictor.parameters())}")
    except Exception as e:
        print(f"Error loading C Predictor weights: {e}")
        sys.exit(1)
    
    hv_predictor.eval()
    for param in hv_predictor.parameters():
        param.requires_grad = False
    print("HV Predictor weights loaded and frozen")

    # Freeze C Predictor
    c_predictor.eval()
    for param in c_predictor.parameters():
        param.requires_grad = False
    print("C Predictor weights loaded and frozen")

    # Initialize strategy generators
    strategy1 = Strategy1Generator(
        hv_predictor, 
        feature_mask_ratio=args.feature_mask_ratio, 
        edge_keep_ratio=args.edge_keep_ratio,
        k_predictor=args.k
    )
    strategy2 = Strategy2Generator(
        c_predictor, 
        feature_mask_ratio=args.feature_mask_ratio, 
        edge_keep_ratio=args.edge_keep_ratio,
        k_predictor=args.k
    )

    print("\n=== Starting GCL model training ===")
    model = GCLModel(
        feat_dim=args.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        tau=args.tau
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.num_epochs):
        with torch.no_grad():
            g1, _ = strategy1.generate_subgraph(data, k=args.k)
            C = compute_propagation_control(data, T=args.T)
            g2 = strategy2.generate_subgraph(data, C, k=args.k)

        u, v = model(g1, g2)
        loss = model.contrastive_loss(u, v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_without_improvement = 0
            torch.save(model.state_dict(), gcl_save_path)
            print(f"Epoch {epoch} | Saved new best model, Loss: {loss.item():.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
            break
        
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Best Loss: {best_loss:.4f}")

    print(f"\nGCL training complete, best loss: {best_loss:.4f}")
    print(f"Best model saved to: {gcl_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GCL Model')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='soc_epinions', help='Dataset name')
    parser.add_argument('--num_features', type=int, default=32, help='Node feature dimension')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--out_dim', type=int, default=64, help='Output dimension')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--print_every', type=int, default=10, help='Print frequency')
    
    # Subgraph generation parameters
    parser.add_argument('--k', type=int, default=10, help='Subgraph size')
    parser.add_argument('--T', type=int, default=5, help='Propagation control steps')
    parser.add_argument('--feature_mask_ratio', type=float, default=0.2, help='Feature mask ratio')
    parser.add_argument('--edge_keep_ratio', type=float, default=0.8, help='Edge keep ratio')

    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=100, help='Epochs to wait without improvement before stopping')

    # Pretrained model paths
    parser.add_argument('--hv_weights_path', type=str, default='pretrained/hv_predictor_best.pth', help='Path to pretrained HV Predictor weights')
    parser.add_argument('--c_weights_path', type=str, default='pretrained/c_predictor_best.pth', help='Path to pretrained C Predictor weights')
    parser.add_argument('--gcl_save_path', type=str, default='', help='Path to save GCL model (default: gcl_models/<dataset>_best_gcl_model.pth)')

    args = parser.parse_args()
    
    start_time = time.time()
    train_integrated(args)
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

