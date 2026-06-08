import torch
import torch.nn as nn
import random
from collections import deque
from GCL import GCLModel
from ic3 import compute_beta_activation, compute_gamma_directed_from_data
from torch_geometric.data import Data
from data_loader import *
from weight_calculate import *
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import argparse
import time 
import pandas as pd
import os
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
cudnn.benchmark = True  


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False
# ----------------------------

class DDQN(nn.Module):
    def __init__(self, graph_data, feat_dim, hidden_dim, out_dim, qnet_hidden_dim, gcl_model_path, name=""):
        super().__init__()
        self.gnn = GCLModel(feat_dim, hidden_dim, out_dim).to(device)
        try:
            self.gnn.load_state_dict(torch.load(gcl_model_path, map_location=device, weights_only=True))
            if name:
                print(f"Successfully loaded {name} GCL model weights: {gcl_model_path}")
        except FileNotFoundError:
            print(f"Warning: GCL model weights file not found: {gcl_model_path}")
        except Exception as e:
            print(f"Error loading GCL model weights: {e}")

        self.q_net = nn.Sequential(
            nn.Linear(out_dim, qnet_hidden_dim),
            nn.ReLU(),
            nn.Linear(qnet_hidden_dim, 1)
        ).to(device)
        self.graph_data = graph_data
        
        self.cached_embeddings = None
        self.cache_valid = False

    def compute_embeddings(self):
        self.gnn.eval()
        with torch.no_grad():
            embeddings = self.gnn.get_embeddings(
                self.graph_data.to(device), 
                self.graph_data.to(device)
            )[0]
        self.cached_embeddings = embeddings
        self.cache_valid = True
        return embeddings

    def forward(self, state):
        
        if not self.cache_valid:
            self.compute_embeddings()
        
        q_values = self.q_net(self.cached_embeddings).squeeze()

        if state.dim() == 2:
            batch_size = state.shape[0]
            return q_values.unsqueeze(0).expand(batch_size, -1).to(device)
        return q_values.to(device)

    def invalidate_cache(self):
        
        self.cache_valid = False

class DDQNAgent:
    def __init__(self, graph_data, discount_factor, epsilon, lr, replay_buffer_size, omega_val, 
                 feat_dim, hidden_dim, out_dim, qnet_hidden_dim, gcl_model_path,
                 use_monte_carlo=False, mc_simulations=100, 
                 epsilon_decay=0.9, epsilon_min=0.01, enable_gcl=False, gcl_update_freq=15):
        self.graph_data = graph_data.to(device)
        self.num_nodes = graph_data.num_nodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.omega_val = omega_val
        
       
        try:
            self.network_gamma = compute_gamma_directed_from_data(self.graph_data)
        except Exception as e_gpu:
            print(f"Error computing gamma on GPU: {e_gpu}, falling back to CPU...")
            try:
                graph_data_cpu = self.graph_data.to('cpu')
                self.network_gamma = compute_gamma_directed_from_data(graph_data_cpu)
            except Exception as e_cpu:
                print(f"Error computing gamma on CPU as well: {e_cpu}")

       

        
        self.reward_cache = {}
        self.edge_index = graph_data.edge_index.to(device)
        self.edge_weight = graph_data.edge_weight.to(device)
        self.active_tensor = torch.zeros(self.num_nodes, device=device)

        
        self.online_net = DDQN(
            self.graph_data, feat_dim, hidden_dim, out_dim, 
            qnet_hidden_dim, gcl_model_path, name="online"
        ).to(device)
        
        self.target_net = DDQN(
            self.graph_data, feat_dim, hidden_dim, out_dim, 
            qnet_hidden_dim, gcl_model_path, name="target"
        ).to(device)
        
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
       
        self.optimizer = torch.optim.Adam(self.online_net.q_net.parameters(), lr=lr)
        
       
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.mc_time = 0.0
        
        
        self.enable_gcl = enable_gcl
        self.gcl_update_freq = gcl_update_freq
        

        self.use_monte_carlo = use_monte_carlo
        self.mc_simulations = mc_simulations

        
        self.states_buffer = torch.zeros((replay_buffer_size, graph_data.num_nodes), device=device)
        self.next_states_buffer = torch.zeros((replay_buffer_size, graph_data.num_nodes), device=device)
        self.actions_buffer = torch.zeros(replay_buffer_size, dtype=torch.long, device=device)
        self.rewards_buffer = torch.zeros(replay_buffer_size, device=device)
        self.dones_buffer = torch.zeros(replay_buffer_size, device=device)
        self.buffer_idx = 0
        self.buffer_full = False

    def compute_reward_vectorized(self, state, action):
        
        with torch.amp.autocast('cuda'):
            active_nodes = state.nonzero().flatten()  
            state_tensor = state.clone()  
            state_tensor[action] = True   
            
            
            prev_reward, _, _, _ = self.propagation_simulation_gpu(state)
            cur_reward, _, _, _ = self.propagation_simulation_gpu(state_tensor)
            return cur_reward - prev_reward
    
    def propagation_simulation_gpu(self, state_tensor):
        
        omega_i = torch.full((self.num_nodes,), self.omega_val, device=device)
        
        
        initial_active = state_tensor.nonzero().flatten()
        
        beta, active = compute_beta_activation(
            initial_Q=initial_active, 
            gamma=self.network_gamma,
            data=self.graph_data,
            omega_i=omega_i
        )
        
        result = active.sum()
        return result.item(), 1, result.item(), [result.item()]

    
    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
    
        
        batch = random.sample(self.replay_buffer, batch_size)
        
       
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.stack(states).to(device)
        next_states_tensor = torch.stack(next_states).to(device)
        actions_tensor = torch.tensor(actions, device=device)
        rewards_tensor = torch.tensor(rewards, device=device)
        dones_tensor = torch.tensor(dones, device=device)
    
       
        with torch.no_grad():
           
            next_q_values = self.online_net(next_states_tensor)
            next_actions = torch.argmax(next_q_values, dim=1)
            target_q_next = self.target_net(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_tensor + (1 - dones_tensor.float()) * self.discount_factor * target_q_next
    
        
        self.online_net.train()
        current_q_values = self.online_net(states_tensor)
        current_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

       
        ddqn_loss = nn.MSELoss()(current_q, target_q.detach())
        
        
        total_loss = ddqn_loss
        if hasattr(self, 'enable_gcl') and self.enable_gcl:
           
            h1, h2 = self.online_net.gnn(self.graph_data, self.graph_data)
           
            gcl_loss = self.online_net.gnn.contrastive_loss(h1, h2)
           
            total_loss = ddqn_loss + gcl_loss  

        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        
        print(f"DDQN Loss: {ddqn_loss.item():.4f}")
        if hasattr(self, 'enable_gcl') and self.enable_gcl:
            print(f"GCL Loss: {gcl_loss.item():.4f}")
            print(f"Total Loss: {total_loss.item():.4f}")

        
        print(f"Step Loss: {total_loss.item():.4f}")  

    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state, available_actions, available_tensor):
        
        if random.random() < self.epsilon:
            
            available_tensor = torch.tensor(available_actions, device=device)
            idx = torch.randint(0, len(available_actions), (1,), device=device)
            return available_tensor[idx].item()
        
        with torch.no_grad():
            q_values = self.online_net(state)
            
            mask = torch.full_like(q_values, float('-inf'), device=device)
            mask[available_tensor] = 0
            q_values = q_values + mask
            return torch.argmax(q_values).item()

    def propagation_simulation(self, seed_set):
       
        omega_i = torch.full((self.num_nodes,), self.omega_val, device=device)
        
        beta, active = compute_beta_activation(
            initial_Q=set(seed_set),
            gamma=self.network_gamma,
            data=self.graph_data,
            omega_i=omega_i  
        )
        return float(active.sum().item()), 1, float(active.sum().item()), [float(active.sum().item())]

if __name__ == "__main__":
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run DDQN with optional test mode and batch experiments")
    # Basic parameters
    parser.add_argument('--train_mc_times', type=int, default=1000, help='MC simulation count during RL training')
    parser.add_argument('--eval_mc_times', type=int, default=1000, help='MC simulation count during final evaluation')
    parser.add_argument('--processes_per_gpu', type=int, default=1, help='Number of processes per GPU (default: 4)')
    parser.add_argument('--dataset_name', type=str, default='fb_mich67', help='Name of the dataset to load')
    parser.add_argument('--save_model_path', type=str, default='best_ddqn_model.pth', help='Path to save the best DDQN model')
    parser.add_argument('--feat_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--qnet_hidden_dim', type=int, default=32)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.35)
    parser.add_argument('--lr', type=float, default=0.0035)
    parser.add_argument('--replay_buffer_size', type=int, default=10000)
    parser.add_argument('--omega_val', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--target_update_freq', type=int, default=8)
    parser.add_argument('--use_monte_carlo', action='store_true')    
    parser.add_argument('--epsilon_decay', type=float, default=0.975, help='Epsilon decay factor per episode')
    parser.add_argument('--epsilon_min', type=float, default=0.08, help='Minimum epsilon value')
    parser.add_argument('--gcl_model_path', type=str, default='gcl_model.pth', help='Path to pretrained GCL model weights')
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode with fewer episodes and experiments')
    args = parser.parse_args()

    
    if args.test_mode:
        print("\n=== Test mode enabled ===")
        SEED_NUMS = [10]
        MC_TIMES = 10
        NUM_EXPERIMENTS = 2
        args.num_episodes = 5
        args.batch_size = 4
    else:
        SEED_NUMS = [10, 20, 30, 40, 50]
        MC_TIMES = 10
        NUM_EXPERIMENTS = 1

    RANDOM_SEED = 42
    DATASET_NAME = args.dataset_name
    mode_prefix = 'test_' if args.test_mode else ''
    RESULTS_DIR = os.path.join("results", f"{mode_prefix}{DATASET_NAME}_experiment_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CSV_PATH = os.path.join("results", f"{mode_prefix}{DATASET_NAME}_ddqn_experiment_results.csv")
    os.makedirs("results", exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    results = []
    total_experiments = len(SEED_NUMS) * NUM_EXPERIMENTS
    print(f"Starting batch experiments, total {total_experiments} runs... Dataset: {DATASET_NAME}")

    # Load and prepare data
    graph_data, _ = load_data(DATASET_NAME)
    graph_data.x = torch.randn((graph_data.num_nodes, args.feat_dim * 2), device=device)
    graph_data = degree_calculate(graph_data)
    graph_data.edge_weight = graph_data.edge_attr

    exp_counter = 0
    for seed_size in SEED_NUMS:
        seed_size_results = []
        for exp_idx in tqdm(range(NUM_EXPERIMENTS), desc=f"Experiments for seed size {seed_size}"):
            exp_counter += 1
            print(f"\nExperiment {exp_idx+1}/{NUM_EXPERIMENTS} (progress: {exp_counter}/{total_experiments}), seed size: {seed_size}")
            start_t = time.time()
            try:
                # Set GCL model weights path
                gcl_model_path = args.gcl_model_path

                # Initialize DDQNAgent with variance-reduction and smoothing hyperparameters


                # Initialize agent
                agent = DDQNAgent(
                    graph_data,
                    discount_factor=args.discount_factor,
                    epsilon=args.epsilon,
                    lr=args.lr,
                    replay_buffer_size=args.replay_buffer_size,
                    omega_val=args.omega_val,
                    feat_dim=args.feat_dim,
                    hidden_dim=args.hidden_dim,
                    out_dim=args.out_dim,
                    qnet_hidden_dim=args.qnet_hidden_dim,
                    gcl_model_path=gcl_model_path,
                    use_monte_carlo=True,
                    mc_simulations=args.train_mc_times,  # MC count for training
                    epsilon_decay=args.epsilon_decay,
                    epsilon_min=args.epsilon_min,
                    enable_gcl=True,
                    gcl_update_freq=args.target_update_freq,
                    # removed gcl_weight parameter
                )

                # Initialize reward histories for this experiment
                rewards_history = []  # reset at the start of each experiment
                eval_history = []  # evaluation under greedy rule

                for episode in tqdm(range(args.num_episodes), desc="Episodes"):
                    state = torch.zeros(graph_data.num_nodes, device=device, dtype=torch.bool)
                    available = list(range(graph_data.num_nodes))

                    # Training step
                    for _ in range(seed_size):
                        action = agent.select_action(state, available, torch.tensor(available, device=device))
                        if action is None:
                            break

                        # remove and record previous state
                        available.remove(action)
                        prev_state = state.clone()

                        # Compute reward increment using old state before taking action
                        raw_reward = agent.compute_reward_vectorized(state, action)
                        reward = raw_reward  # use raw reward directly

                        # Update state and record seed
                        state[action] = True

                        # Store experience transition (prev_state, action, reward, next_state)
                        done = False
                        agent.replay_buffer.append((prev_state, action, reward, state.clone(), done))
                        agent.train_step(args.batch_size)

                    # Compute and record reward for this episode
                    ep_reward, _, _, _ = agent.propagation_simulation(set(state.nonzero().flatten().tolist()))
                    rewards_history.append(ep_reward)  # accumulate episode rewards

                    # Evaluation mode
                    old_eps = agent.epsilon
                    old_mc_simulations = agent.mc_simulations
                    agent.epsilon = 0.0
                    agent.mc_simulations = args.eval_mc_times

                    # Record evaluation start time
                    eval_start_time = time.time()

                    # Run evaluation logic
                    eval_state = torch.zeros(graph_data.num_nodes, device=device, dtype=torch.bool)
                    eval_available = list(range(graph_data.num_nodes))
                    eval_selected = []

                    for _ in range(seed_size):
                        eval_action = agent.select_action(eval_state, eval_available, torch.tensor(eval_available, device=device))
                        if eval_action is None:
                            break
                        eval_selected.append(eval_action)
                        eval_available.remove(eval_action)
                        eval_state[eval_action] = True

                    # Run evaluation Monte Carlo simulation
                    eval_reward, _, _, _ = agent.propagation_simulation(set(eval_selected))

                    # Record evaluation end time and compute duration
                    eval_end_time = time.time()
                    eval_mc_time = eval_end_time - eval_start_time

                    # Restore training parameters
                    agent.epsilon = old_eps
                    agent.mc_simulations = old_mc_simulations

                    # Record evaluation reward
                    eval_history.append(eval_reward)


                    # Clear history for the next experiment
                    rewards_history = []  # clear history for next experiment
                    eval_history = []

                    # Restore epsilon and update target network
                    agent.epsilon = old_eps
                    if (episode + 1) % args.target_update_freq == 0:
                        agent.update_target_net()

                    # Epsilon decay
                    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

            except Exception as e:
                print(f"Error initializing or training DDQN Agent: {e}")
                continue


        # Save model checkpoint
        model_path = os.path.join(RESULTS_DIR, f"ddqn_seed{seed_size}_exp{exp_idx+1}.pth")
        # Save model using file handle to support unicode paths on Windows
        with open(model_path, 'wb') as f:
            torch.save(agent.online_net.state_dict(), f)
        # Final greedy evaluation
        final_selected = []
        state = torch.zeros(graph_data.num_nodes, device=device, dtype=torch.bool)
        available = list(range(graph_data.num_nodes))
        agent.epsilon = 0
        for _ in range(seed_size):
            action = agent.select_action(state, available, torch.tensor(available, device=device))
            if action is None: break
            final_selected.append(action)
            available.remove(action)
            state[action] = True

        # Print selected seed nodes for this experiment
        print(f"Experiment {exp_idx+1} seed size {seed_size} selected nodes: {sorted(final_selected)}")

        # Record runtime excluding Monte Carlo simulation time
        end_t = time.time()
        run_time = end_t - start_t - eval_mc_time - agent.mc_time  # subtract all MC time

        # Monte Carlo average activation
        avg_act, _, _, _ = agent.propagation_simulation(set(final_selected))
        # Deterministic expected activation (via propagation_simulation)
        det_act, _, _, _ = agent.propagation_simulation(set(final_selected))
        print(f"Total time: {end_t - start_t:.2f}s")
        print(f"Evaluation time: {eval_mc_time:.2f}s")
        print(f"Runtime excluding evaluation: {run_time:.2f}s")
        results.append({
            'seed_size': seed_size,
            'experiment': exp_idx+1,
            'average_mc': float(avg_act),
            'deterministic': float(det_act),
            'Selected Seeds': sorted(final_selected),
            'model_path': model_path,
            'run_time_need': run_time,  # excluding evaluation time
            'eval_time_not_need': eval_mc_time,
            'eval_reward': eval_reward,  # diffusion score for this experiment
        })
        seed_size_results.append(results[-1])

        # Print summary
        if seed_size_results:
            acts = [r['average_mc'] for r in seed_size_results]
            print(f"Seed {seed_size} average activation: {avg_act} ")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print("All experiments completed! Results saved to", CSV_PATH)

    
