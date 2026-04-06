import argparse
import networkx as nx
import torch
import numpy as np
import math
import time
from tqdm import trange
from numba import njit

try:
    from fast_sampler import sample_non_edges_directed_numba, sample_non_edges_undirected_numba
    print("Successfully imported Numba fast sampling")
    USE_FAST_SAMPLING = True
except ImportError:
    print("Numba or fast_sampler.py not available. Using pure Python sampling.")
    USE_FAST_SAMPLING = False

# ------------------------------------------------------------------ #
# NUMBA C-SPEED MCMC ENGINE (LOG-SPACE OPTIMIZED)                    #
# ------------------------------------------------------------------ #
@njit(fastmath=True)
def get_kron_log_prob(u_slot, v_slot, k, lp00, lp01, lp10, lp11):
    """Calculates the Log-Probability natively to prevent underflow."""
    log_prob = 0.0
    for idx in range(k):
        # Shift bits from highest (k-1) to lowest (0)
        bit_u = (u_slot >> ((k - 1) - idx)) & 1
        bit_v = (v_slot >> ((k - 1) - idx)) & 1
        if bit_u == 0:
            if bit_v == 0: log_prob += lp00
            else: log_prob += lp01
        else:
            if bit_v == 0: log_prob += lp10
            else: log_prob += lp11
            
    return log_prob

@njit(fastmath=True)
def run_mcmc_numba(steps, n_nodes, n_edges, k, lp00, lp01, lp10, lp11,
                   perm, inv_perm, edge_list, node_offsets, node_edges):
    accepted = 0
    for step in range(steps):
        # 1. Propose Swap (20% Random Node, 80% Random Edge)
        if np.random.rand() < 0.2:
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i == j: continue
            pi = perm[i]
            pj = perm[j]
        else:
            e_idx = np.random.randint(0, n_edges)
            pi = edge_list[e_idx, 0]
            pj = edge_list[e_idx, 1]
            if pi == pj: continue
            i = inv_perm[pi]
            j = inv_perm[pj]

        ll_old = 0.0
        ll_new = 0.0

        # Evaluate edges touching node 'pi'
        start_i = node_offsets[pi]
        end_i = node_offsets[pi+1]
        for ptr in range(start_i, end_i):
            e_idx = node_edges[ptr]
            u_node = edge_list[e_idx, 0]
            v_node = edge_list[e_idx, 1]

            # Old LL (Log-Space)
            u_slot_old = inv_perm[u_node]
            v_slot_old = inv_perm[v_node]
            log_prob_old = get_kron_log_prob(u_slot_old, v_slot_old, k, lp00, lp01, lp10, lp11)
            prob_old = math.exp(log_prob_old) # Safe to exponentiate here for the taylor term
            ll_old += log_prob_old - (-prob_old - 0.5 * prob_old**2)

            # New LL 
            u_slot_new = j if u_node == pi else (i if u_node == pj else inv_perm[u_node])
            v_slot_new = j if v_node == pi else (i if v_node == pj else inv_perm[v_node])
            log_prob_new = get_kron_log_prob(u_slot_new, v_slot_new, k, lp00, lp01, lp10, lp11)
            prob_new = math.exp(log_prob_new)
            ll_new += log_prob_new - (-prob_new - 0.5 * prob_new**2)

        # Evaluate edges touching node 'pj'
        start_j = node_offsets[pj]
        end_j = node_offsets[pj+1]
        for ptr in range(start_j, end_j):
            e_idx = node_edges[ptr]
            u_node = edge_list[e_idx, 0]
            v_node = edge_list[e_idx, 1]

            if (u_node == pi and v_node == pj) or (u_node == pj and v_node == pi):
                continue

            # Old LL
            u_slot_old = inv_perm[u_node]
            v_slot_old = inv_perm[v_node]
            log_prob_old = get_kron_log_prob(u_slot_old, v_slot_old, k, lp00, lp01, lp10, lp11)
            prob_old = math.exp(log_prob_old)
            ll_old += log_prob_old - (-prob_old - 0.5 * prob_old**2)

            # New LL
            u_slot_new = j if u_node == pi else (i if u_node == pj else inv_perm[u_node])
            v_slot_new = j if v_node == pi else (i if v_node == pj else inv_perm[v_node])
            log_prob_new = get_kron_log_prob(u_slot_new, v_slot_new, k, lp00, lp01, lp10, lp11)
            prob_new = math.exp(log_prob_new)
            ll_new += log_prob_new - (-prob_new - 0.5 * prob_new**2)

        delta = ll_new - ll_old
        if delta > 0 or np.random.rand() < math.exp(delta):
            perm[i] = pj
            perm[j] = pi
            inv_perm[pi] = j
            inv_perm[pj] = i
            accepted += 1

    return accepted

# ------------------------------------------------------------------ #
# MAIN PYTORCH CLASS                                                 #
# ------------------------------------------------------------------ #
class mfit:
    def __init__(
        self,
        graph_temp,
        init_matrix,
        learning_rate=0.05,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.is_directed = False
        for u, v in graph_temp.edges():
            if not graph_temp.has_edge(v, u):
                self.is_directed = True
                break
    
        if self.is_directed:
            graph_raw = graph_temp
        else:
            graph_raw = graph_temp.to_undirected()
            print(f"Graph Mode: {'Directed' if self.is_directed else 'Undirected'} (Auto-Detected)")
        
        self.graph = nx.convert_node_labels_to_integers(graph_raw, first_label=0, ordering="default")
        self.init_matrix = np.array(init_matrix).reshape(2, 2)

        self.n_nodes = self.graph.number_of_nodes()
        self.k = int(np.ceil(np.log2(self.n_nodes)))
        
        padded_n_nodes = 2 ** self.k
        if padded_n_nodes != self.n_nodes:
            print(f"Padding graph from {self.n_nodes} nodes to {padded_n_nodes} for model")
            self.graph.add_nodes_from(range(self.n_nodes, padded_n_nodes))
            self.n_nodes = padded_n_nodes
        
        self.n_edges = self.graph.number_of_edges()
        
        self.edge_list_tensor = torch.tensor(list(self.graph.edges()), dtype=torch.long, device=self.device)
        self.node_to_edges = {n: [] for n in range(self.n_nodes)}
        for idx, (u, v) in enumerate(self.edge_list_tensor.tolist()):
            self.node_to_edges[u].append(idx)
            self.node_to_edges[v].append(idx)
            
        self.numba_edge_list = self.edge_list_tensor.cpu().numpy().astype(np.int32)
        offsets = np.zeros(self.n_nodes + 1, dtype=np.int32)
        for n in range(self.n_nodes):
            offsets[n+1] = offsets[n] + len(self.node_to_edges[n])
        self.numba_node_edges = np.zeros(offsets[-1], dtype=np.int32)
        for n in range(self.n_nodes):
            start, end = offsets[n], offsets[n+1]
            self.numba_node_edges[start:end] = self.node_to_edges[n]
        self.numba_node_offsets = offsets
        
        scaled_p_np = self._get_scaled_initial_matrix(self.init_matrix)
        print("Initial initiator matrix (scaled):\n", scaled_p_np)
        
        self.p00 = torch.nn.Parameter(torch.tensor(scaled_p_np[0, 0], dtype=torch.float32, device=self.device))
        self.p01 = torch.nn.Parameter(torch.tensor(scaled_p_np[0, 1], dtype=torch.float32, device=self.device))
        self.p10 = torch.nn.Parameter(torch.tensor(scaled_p_np[1, 0], dtype=torch.float32, device=self.device))
        self.p11 = torch.nn.Parameter(torch.tensor(scaled_p_np[1, 1], dtype=torch.float32, device=self.device))
        
        self.optimizer = torch.optim.Adam([
            {'params': self.p00, 'lr': learning_rate},
            {'params': self.p01, 'lr': learning_rate},
            {'params': self.p10, 'lr': learning_rate},
            {'params': self.p11, 'lr': learning_rate},
        ], betas=(0.5, 0.9))

        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        self.perm_np = np.array(sorted_nodes, dtype=np.int32)
        
        inv = np.empty_like(self.perm_np)
        inv[self.perm_np] = np.arange(self.n_nodes, dtype=np.int32)
        self.inv_perm_np = inv

        self.perm = torch.tensor(self.perm_np, dtype=torch.long, device=self.device)
        self.inverse_perm = torch.tensor(self.inv_perm_np, dtype=torch.long, device=self.device)

    def _get_scaled_initial_matrix(self, init_matrix):
        p_np = np.array(init_matrix, dtype=np.float64)
        if self.n_edges > 0:
            expected_edges = np.sum(p_np) ** self.k
            if expected_edges > 0:
                scale_factor = (self.n_edges / expected_edges) ** (1.0 / self.k) if self.k > 0 else 1.0
                p_np *= scale_factor
        return np.clip(p_np, 1e-4, 1.0 - 1e-4)

    def _calc_total_ll(self):
        sum_P = self.p00 + self.p01 + self.p10 + self.p11
        sum_P2 = self.p00**2 + self.p01**2 + self.p10**2 + self.p11**2
        ll_empty = -(sum_P ** self.k) - 0.5 * (sum_P2 ** self.k)

        perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
        perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]
        
        # --------------------------------------------------------- #
        # INTEGER BIT-COUNTING TRICK (Native Log-Space)             #
        # --------------------------------------------------------- #
        with torch.no_grad():
            n11 = torch.zeros_like(perm_u, dtype=torch.int8)
            n10 = torch.zeros_like(perm_u, dtype=torch.int8)
            n01 = torch.zeros_like(perm_u, dtype=torch.int8)
            
            for bit in range(self.k):
                shift = (self.k - 1) - bit
                u_b = (perm_u >> shift) & 1
                v_b = (perm_v >> shift) & 1
                
                n11 += (u_b & v_b).to(torch.int8)
                n10 += (u_b & (1 - v_b)).to(torch.int8)
                n01 += ((1 - u_b) & v_b).to(torch.int8)
                
            n00 = self.k - n11 - n10 - n01

        # Re-attach to computation graph securely in log-space!
        log_p_e = (n00 * torch.log(self.p00) + 
                   n01 * torch.log(self.p01) + 
                   n10 * torch.log(self.p10) + 
                   n11 * torch.log(self.p11))
                 
        p_e = torch.exp(log_p_e) # Safe exponentiation for the approximation
        
        # Because log_p_e is already the exact logarithm, we don't need torch.log()
        # and we don't need to clamp!
        ll_edges_actual = log_p_e
        ll_edges_fake_no_edge = -p_e - 0.5 * (p_e ** 2)

        return ll_empty + torch.sum(ll_edges_actual - ll_edges_fake_no_edge)

    def fit(self, iterations, warmup_mcmc, mcmc_per_iter):
        print(f"\nMCMC Numba Warm-up ({warmup_mcmc} steps)")
        
        # Pre-calculate logarithms to pass into Numba
        lp00, lp01 = math.log(self.p00.item()), math.log(self.p01.item())
        lp10, lp11 = math.log(self.p10.item()), math.log(self.p11.item())
        
        accepted_swaps = run_mcmc_numba(
            warmup_mcmc, self.n_nodes, self.n_edges, self.k,
            lp00, lp01, lp10, lp11,
            self.perm_np, self.inv_perm_np, self.numba_edge_list,
            self.numba_node_offsets, self.numba_node_edges
        )
        print(f"Warm-up acceptance rate: {accepted_swaps/(warmup_mcmc or 1):.2%}")

        print(f"\nMain Optimization ({iterations} iterations)")
        best_ll = float('-inf')
        best_P = [self.p00.item(), self.p01.item(), self.p10.item(), self.p11.item()]

        num_chunks = 10
        chunk_size = max(1, mcmc_per_iter // num_chunks)

        for iteration in range(iterations):
            iter_start_time = time.time()
            self.optimizer.zero_grad()
            
            mcmc_accepted = 0
            avg_ll = 0
            grad_evals = 0
            
            # Recalculate logs for the current parameters
            lp00, lp01 = math.log(self.p00.item()), math.log(self.p01.item())
            lp10, lp11 = math.log(self.p10.item()), math.log(self.p11.item())
            
            for chunk in range(num_chunks):
                mcmc_accepted += run_mcmc_numba(
                    chunk_size, self.n_nodes, self.n_edges, self.k,
                    lp00, lp01, lp10, lp11,
                    self.perm_np, self.inv_perm_np, self.numba_edge_list,
                    self.numba_node_offsets, self.numba_node_edges
                )
                
                self.perm.data.copy_(torch.from_numpy(self.perm_np).to(self.device))
                self.inverse_perm.data.copy_(torch.from_numpy(self.inv_perm_np).to(self.device))
                
                ll = self._calc_total_ll()
                loss = -ll
                loss.backward()
                avg_ll += ll.item()
                grad_evals += 1
            
            for param in [self.p00, self.p01, self.p10, self.p11]:
                if param.grad is not None:
                    param.grad /= grad_evals
            
            current_ll = avg_ll / grad_evals
            self.optimizer.step()

            with torch.no_grad():
                self.p00.clamp_(1e-4, 0.9999)
                self.p01.clamp_(1e-4, 0.9999)
                self.p10.clamp_(1e-4, 0.9999)
                self.p11.clamp_(1e-4, 0.9999)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.98

            if current_ll > best_ll:
                best_ll = current_ll
                best_P = [self.p00.item(), self.p01.item(), self.p10.item(), self.p11.item()]

            elapsed = time.time() - iter_start_time
            
            print(
                f"\n{iteration+1:3d}/{iterations}] "
                f"LL: {current_ll:12.2f}  Best LL: {best_ll:12.2f}  "
                f"Wu: {mcmc_accepted / max(mcmc_per_iter, 1):5.1%}  "
                f"Time: {elapsed:.1f}s"
            )
            print(
                f"  P = [{self.p00.item():.4f}, {self.p01.item():.4f}]\n"
                f"      [{self.p10.item():.4f}, {self.p11.item():.4f}]"
            )
            print(
                f"  lr = [{self.optimizer.param_groups[0]['lr']:.5f}, "
                f"{self.optimizer.param_groups[1]['lr']:.5f}, "
                f"{self.optimizer.param_groups[2]['lr']:.5f}, "
                f"{self.optimizer.param_groups[3]['lr']:.5f}]"
            )

        print(f"\nBest P (LL={best_ll:.2f}):")
        print(f"  [{best_P[0]:.4f}, {best_P[1]:.4f}]")
        print(f"  [{best_P[2]:.4f}, {best_P[3]:.4f}]")
        return best_P, best_ll

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("--init_matrix", nargs=4, type=float, default=[0.9, 0.7, 0.5, 0.2])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warmup_mcmc", type=int, default=10000)
    parser.add_argument("--grad_samples", type=int, default=100000)
    args = parser.parse_args()
    start_time = time.time()

    print("Loading and preparing graph...")
    graph = nx.read_edgelist(args.file_path, nodetype=int, create_using=nx.DiGraph(), comments="#")
    
    fitter = mfit(graph, args.init_matrix, learning_rate=args.lr)
    best_P, best_ll = fitter.fit(iterations=args.iterations, warmup_mcmc=args.warmup_mcmc, mcmc_per_iter=args.grad_samples)
    
    print(f"\nTotal Execution Time: {time.time()-start_time:.2f} seconds")
    print(f"\nBest P (LL={best_ll:.2f}):")
    print(f"  [{best_P[0]:.4f}, {best_P[1]:.4f}]")
    print(f"  [{best_P[2]:.4f}, {best_P[3]:.4f}]")


if __name__ == "__main__":
    main()