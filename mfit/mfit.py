import torch
import numpy as np
import random
from tqdm import trange
import time
import networkx as nx

class mfit:
    def __init__(self, graph_temp, init_matrix, learning_rate=1e-5, warmup_mcmc=10000, grad_samples=100000, iterations=50, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(0))
        
        # 1. Graph Setup
        self.is_directed = graph_temp.is_directed()
        graph_raw = nx.convert_node_labels_to_integers(graph_temp, first_label=0)
        self.n_nodes = graph_raw.number_of_nodes()
        
        # 2. Kronecker Dimensions
        self.k = int(np.ceil(np.log2(self.n_nodes)))
        self.padded_n_nodes = 2 ** self.k
        if self.padded_n_nodes > self.n_nodes:
            graph_raw.add_nodes_from(range(self.n_nodes, self.padded_n_nodes))
            self.n_nodes = self.padded_n_nodes
            
        graph = graph_raw
        self.n_edges = graph.number_of_edges()
        self.edge_list_tensor = torch.tensor(list(graph.edges()), dtype=torch.long, device=self.device)
        
        # Edge vector for "EdgeSwap" MCMC proposals (matching GEdgeV in SNAP)
        self.edge_vector = self.edge_list_tensor.clone()
        
        # 3. Parameters (Initiator Matrix)
        # SNAP default init often scales to match edge count
        p_init = np.array(init_matrix).reshape(2, 2)
        current_sum = np.sum(p_init)
        target_sum = self.n_edges ** (1.0 / self.k)
        p_init = p_init * (target_sum / current_sum)
        p_init = np.clip(p_init, 0.001, 0.999)
        
        # Use a plain tensor for manual gradient updates
        self.P = torch.tensor(p_init, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 4. Permutation Setup
        degrees = dict(graph.degree())
        self.perm = self._initialize_permutation_by_degree(degrees)
        self.inverse_perm = torch.empty_like(self.perm)
        self.inverse_perm[self.perm] = torch.arange(self.n_nodes, device=self.device)
        
        # 5. Training Hyperparams
        self.lr = learning_rate
        self.warmup_mcmc = warmup_mcmc
        self.grad_samples = grad_samples
        self.iterations = iterations
        self.mn_step = 0.005
        self.mx_step = 0.05
        self.perm_swap_node_prob = 0.2 # SNAP default: swap random nodes 20% of time, edges 80%

    def _initialize_permutation_by_degree(self, degrees):
        
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        return torch.tensor(sorted_nodes, dtype=torch.long, device=self.device)

    def _get_kron_edge_prob(self, u_idx, v_idx):
        # Vectorized bit-extraction matching SNAP level logic
        k_range = torch.arange(self.k - 1, -1, -1, device=self.device)
        u_bits = (u_idx.unsqueeze(-1) >> k_range) & 1
        v_bits = (v_idx.unsqueeze(-1) >> k_range) & 1
        
        # Map bits to matrix indices
        # Bit 0,0 -> P[0,0]; Bit 0,1 -> P[0,1]...
        probs_levels = torch.where(u_bits == 0, 
                                   torch.where(v_bits == 0, self.P[0,0], self.P[0,1]),
                                   torch.where(v_bits == 0, self.P[1,0], self.P[1,1]))
        return torch.prod(probs_levels, dim=1)

    def _get_apx_empty_graph_ll(self):
        # Matching SNAP: -sum(theta)^k - 0.5 * sum(theta^2)^k
        sum_p = torch.sum(self.P)
        sum_p2 = torch.sum(self.P**2)
        return -(sum_p**self.k) - 0.5 * (sum_p2**self.k)

    def _calculate_ll(self):
        # L = LL_empty + sum_{edges}(LL_edge(u,v) - LL_apx_no_edge(u,v))
        ll_empty = self._get_apx_empty_graph_ll()
        
        u_perm = self.inverse_perm[self.edge_list_tensor[:, 0]]
        v_perm = self.inverse_perm[self.edge_list_tensor[:, 1]]
        
        edge_probs = self._get_kron_edge_prob(u_perm, v_perm).clamp(1e-10, 0.9999)
        
        # edge_ll = log(p)
        ll_edges = torch.log(edge_probs).sum()
        # apx_no_edge_ll = -p - 0.5p^2
        ll_apx_no_edge = (-edge_probs - 0.5 * (edge_probs**2)).sum()
        
        return ll_empty + ll_edges - ll_apx_no_edge

    def _mcmc_step(self):
        # SNAP logic: 20% random node swap, 80% random edge endpoint swap
        if random.random() < self.perm_swap_node_prob:
            i, j = random.sample(range(self.n_nodes), 2)
        else:
            e_idx = random.randint(0, self.n_edges - 1)
            i, j = self.edge_vector[e_idx].tolist()

        if i == j: return False
        
        old_ll = self._calculate_ll()
        
        # Swap
        node_i, node_j = self.perm[i].item(), self.perm[j].item()
        self.perm[i], self.perm[j] = node_j, node_i
        self.inverse_perm[node_i], self.inverse_perm[node_j] = j, i
        
        new_ll = self._calculate_ll()
        
        # Metropolis Accept/Reject
        if new_ll > old_ll or random.random() < torch.exp(new_ll - old_ll).item():
            return True
        else:
            # Revert
            self.perm[i], self.perm[j] = node_i, node_j
            self.inverse_perm[node_i], self.inverse_perm[node_j] = i, j
            return False

    def fit(self):
        start_time = time.time()
        learn_rate_v = torch.full((4,), self.lr, device=self.device)
        best_ll = -float('inf')
        
        print("Starting Warmup...")
        for _ in trange(self.warmup_mcmc):
            self._mcmc_step()

        print("Starting Gradient Descent...")
        for iter_idx in range(self.iterations):
            # 1. Sample Gradient (SNAP style)
            # We use autograd on our LL function to get the gradient
            current_ll = self._calculate_ll()
            current_ll.backward()
            
            grad = self.P.grad.flatten()
            
            # 2. Update Parameters manually (Match SNAP's GradDescent logic)
            with torch.no_grad():
                p_flat = self.P.flatten()
                for p in range(4):
                    # Dynamic learning rate adjustment
                    move = learn_rate_v[p] * grad[p]
                    while abs(move) > self.mx_step:
                        learn_rate_v[p] *= 0.95
                        move = learn_rate_v[p] * grad[p]
                    while abs(move) < self.mn_step:
                        learn_rate_v[p] *= (1.0/0.95)
                        move = learn_rate_v[p] * grad[p]
                    
                    p_flat[p] += move
                
                self.P.copy_(p_flat.view(2, 2).clamp(0.0001, 0.9999))
                self.P.grad.zero_()

            # 3. MCMC Permutation Update
            accepted = 0
            for _ in range(self.grad_samples // 100): # Sample subset per iteration
                if self._mcmc_step(): accepted += 1

            print(f"Iter {iter_idx+1}: LL {current_ll.item():.2f} | P: {self.P.flatten().tolist()}")
        total_time = time.time() - start_time
            
        return {
            "theta": self.P.detach().cpu().numpy().flatten().tolist(),
            "time": total_time
        }