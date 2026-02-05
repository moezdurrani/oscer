import torch
import numpy as np
import random
from tqdm import trange
import time
import networkx as nx

try:
    from fast_sampler import sample_non_edges_directed_numba, sample_non_edges_undirected_numba
    print("Successfully imported Numba fast sampling")
    USE_FAST_SAMPLING = True
except ImportError:
    print("Numba or fast_sampler.py not available. Using pure Python sampling.")
    USE_FAST_SAMPLING = False

class mfit:
    def __init__(self, graph_temp, init_matrix, learning_rate, warmup_mcmc, grad_samples, iterations, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        
        self.is_directed = False
        for u, v in graph_temp.edges():
            if not graph_temp.has_edge(v, u):
                self.is_directed = True
                break # Found an asymmetric edge, so it's definitely directed.
    
        # 3. Finalize the graph object. If it was symmetric, convert to undirected for efficiency.
        if self.is_directed:
            graph_raw = graph_temp
        else:
            graph_raw = graph_temp.to_undirected()
            print(f"Graph Mode: {'Directed' if self.is_directed else 'Undirected'} (Auto-Detected)")
        
        # remapping labels to clean number sequence
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

        # Build node → edge index map for fast delta likelihood updates
        self.node_to_edges = {n: [] for n in range(self.n_nodes)}
        for idx, (u, v) in enumerate(self.edge_list_tensor.tolist()):
            self.node_to_edges[u].append(idx)
            self.node_to_edges[v].append(idx)

        if self.is_directed:
            self.edge_set = {tuple(edge) for edge in self.graph.edges()}
            total_possible_pairs = self.n_nodes * (self.n_nodes - 1)
        else:
            self.edge_set = {tuple(sorted(edge)) for edge in self.graph.edges()}
            total_possible_pairs = self.n_nodes * (self.n_nodes - 1) // 2

        self.num_non_edges = total_possible_pairs - len(self.edge_set)

        self.edge_array_for_numba = None
        if USE_FAST_SAMPLING:
            print("Preparing edge array for Numba Fast Sampling")
            self.edge_array_for_numba = np.array(list(self.edge_set), dtype=np.int32)
            print("Edge array is ready")
        
        scaled_p_np = self._get_scaled_initial_matrix(self.init_matrix)
        print("Initial initiator matrix (scaled):\n", scaled_p_np)
        
        self.P = torch.nn.Parameter(torch.tensor(scaled_p_np, dtype=torch.float32, device=self.device))
        self.perm = self._initialize_permutation_by_degree()
        # perm[0] = 2 (slot 0 holds node 2) slot -> node
        # inverse_perm[2] = 0 (node 2 is at slot 0) node -> slot
        self.inverse_perm = self._calculate_inverse_perm()
        self.optimizer = torch.optim.Adam([self.P], lr=learning_rate)

        # total = warmup samples + (iterations * grad_samples)
        self.warmup_mcmc = warmup_mcmc
        self.grad_samples = grad_samples
        self.iterations = iterations
        total_needed = self.warmup_mcmc + (self.iterations * grad_samples)
        total_needed = int(total_needed * 1.1)  # add buffer

        print(f"\n[Precomputing {total_needed:,} non-edges for training]")
        self.all_non_edges = self._sample_non_edges(total_needed)

        # Split into warm-up and train sets
        self.warmup_non_edges = self.all_non_edges[:warmup_mcmc]
        self.train_non_edges = self.all_non_edges[warmup_mcmc:]
        self.non_edge_ptr = 0

        self.edge_ll = None
        self.non_edge_ll = None
        self.sampled_non_edges = None

    def _initialize_likelihood_cache(self, non_edge_sample):
        """Precompute per-edge and per-non-edge log-likelihood contributions."""
        # edges
        perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
        perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]
        probs_e = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
        self.edge_ll = torch.log(probs_e.clamp(min=1e-10))

        # non-edges
        perm_u = self.inverse_perm[non_edge_sample[:, 0]]
        perm_v = self.inverse_perm[non_edge_sample[:, 1]]
        probs_ne = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
        scale = self.num_non_edges / len(non_edge_sample)
        self.non_edge_ll = torch.log((1 - probs_ne).clamp(min=1e-10)) * scale

        self.current_ll = self.edge_ll.sum() + self.non_edge_ll.sum()
        self.sampled_non_edges = non_edge_sample.clone()

    def _delta_ll_for_swap(self, u, v):
        """Compute the change in log-likelihood caused by swapping nodes u and v."""
        affected_edge_idx = list(set(self.node_to_edges[u] + self.node_to_edges[v]))
        affected_edges = self.edge_list_tensor[affected_edge_idx]

        # Subtract old contributions
        delta = -self.edge_ll[affected_edge_idx].sum()

        # Recompute with updated permutation
        perm_u = self.inverse_perm[affected_edges[:, 0]]
        perm_v = self.inverse_perm[affected_edges[:, 1]]
        new_probs = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
        new_ll = torch.log(new_probs.clamp(min=1e-10))

        # Add new contributions
        delta += new_ll.sum()

        # Update cache tensors
        self.edge_ll[affected_edge_idx] = new_ll

        return delta


    def _get_scaled_initial_matrix(self, init_matrix):
        p_np = np.array(init_matrix, dtype=np.float64)
        if self.n_edges > 0:
            # avg_p = np.sum(p_np) / 4.0
            # expected_edges = (self.n_nodes ** 2) * (avg_p ** self.k)
            expected_edges = np.sum(p_np) ** self.k
            if expected_edges > 0:
                scale_factor = (self.n_edges / expected_edges) ** (1.0 / self.k) if self.k > 0 else 1.0
                p_np *= scale_factor
        return p_np

    def _initialize_permutation_by_degree(self):
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        return torch.tensor(sorted_nodes, dtype=torch.long, device=self.device)

    def _calculate_inverse_perm(self):
        inverse_perm = torch.empty_like(self.perm)
        inverse_perm[self.perm] = torch.arange(self.n_nodes, device=self.device)
        return inverse_perm

    def _get_kron_edge_prob_vectorized(self, u_indices, v_indices):
        # Create a tensor of bit positions to extract, from highest (k-1) to lowest (0)
        k_range = torch.arange(self.k - 1, -1, -1, device=self.device)

        # Convert node indices into binary bit-strings of length k.
        # 1. unsqueeze(-1) → reshape [n_edges] into [n_edges,1]
        # 2. >> k_range    → shift right by each bit position in k_range.
        # 3. & 1           → mask out all but the last bit, giving 0 or 1.
        # Result: u_bits[i,:] = binary digits of u_indices[i]
        u_bits = (u_indices.unsqueeze(-1) >> k_range) & 1
        v_bits = (v_indices.unsqueeze(-1) >> k_range) & 1

        # Extract scalars from P (still trainable parameters!)
        A = self.P[0, 0]
        B = self.P[0, 1]
        C = self.P[1, 0]
        D = self.P[1, 1]

        # Look up the probability in the initiator matrix P for each (u_bit, v_bit) pair
        probs_k = ((1 - u_bits) * (1 - v_bits) * A +
                    (1 - u_bits) * (v_bits) * B +
                    (u_bits) * (1 - v_bits) * C +
                    (u_bits) * (v_bits) * D)

        return torch.prod(probs_k, dim=1)

    def _sample_non_edges(self, n_samples):
        if USE_FAST_SAMPLING:
            if self.is_directed:
                non_edges_np = sample_non_edges_directed_numba(self.n_nodes, self.edge_array_for_numba, n_samples)
            else:
                non_edges_np = sample_non_edges_undirected_numba(self.n_nodes, self.edge_array_for_numba, n_samples)
            return torch.tensor(non_edges_np, dtype=torch.long, device=self.device)

        non_edges = []
        # Could use a predefined torch tensor with a fixed length
        # non_edges = torch.empty((n_samples, 2), dtype=torch.int64, device=self.device)
        # non_edges[filled] = torch.tensor([u, v], dtype=torch.int64, device=self.device)
        # return non_edges[:filled]

        attempts = 0
        max_attempts = n_samples * 10
        while len(non_edges) < n_samples and attempts < max_attempts:
            u = random.randint(0, self.n_nodes - 1)
            v = random.randint(0, self.n_nodes - 1)
            if u == v: continue
            # ignore self loops

            edge = (u, v) if self.is_directed else tuple(sorted((u, v)))
            if edge in self.edge_set: continue

            non_edges.append((u, v))
            attempts += 1
        return torch.tensor(non_edges, dtype=torch.int64, device=self.device)

    def _calculate_log_likelihood(self, edges_to_eval, non_edges_to_eval):
        # Map edge endpoints from node IDs → permutation slots
        permuted_edges_u = self.inverse_perm[edges_to_eval[:, 0]]
        permuted_edges_v = self.inverse_perm[edges_to_eval[:, 1]]

        # Compute Kronecker probabilities for each edge (vector of size [n_edges])
        prob_edges = self._get_kron_edge_prob_vectorized(permuted_edges_u, permuted_edges_v)

        # Log-likelihood contribution from real edges
        ll_edges = torch.log(prob_edges.clamp(min=1e-10)).sum()

        if len(non_edges_to_eval) > 0:

            # Map sampled non-edges from node IDs → permutation slots
            permuted_non_edges_u = self.inverse_perm[non_edges_to_eval[:, 0]]
            permuted_non_edges_v = self.inverse_perm[non_edges_to_eval[:, 1]]
            prob_non_edges = self._get_kron_edge_prob_vectorized(permuted_non_edges_u, permuted_non_edges_v)

            # Scale factor = total number of non-edges ÷ number sampled
            scale_factor = self.num_non_edges / len(non_edges_to_eval)

            # Log-likelihood contribution from non-edges
            ll_non_edges = torch.log((1 - prob_non_edges).clamp(min=1e-10)).sum() * scale_factor
        else:
            ll_non_edges = 0.0

        return ll_edges + ll_non_edges

    def _mcmc_step_for_permutation(self):
        i, j = torch.randint(0, self.n_nodes, (2,)).tolist()
        if i == j:
            return False

        pi, pj = self.perm[i].item(), self.perm[j].item()
        self.perm[i], self.perm[j] = pj, pi
        self.inverse_perm[pi], self.inverse_perm[pj] = j, i

        # Compute delta using affected edges only
        delta = self._delta_ll_for_swap(pi, pj)

        # Metropolis-Hastings accept/reject
        if delta > 0 or torch.rand(1).to(self.device) < torch.exp(delta):
            self.current_ll += delta
            return True
        else:
            # revert swap
            self.perm[i], self.perm[j] = pi, pj
            self.inverse_perm[pi], self.inverse_perm[pj] = i, j
            return False


    def fit(self, iterations, grad_samples, warmup_mcmc, mcmc_per_iter, verbose=True):
        start_time = time.time()
        best_ll = float("-inf")
        best_theta = None
        
        if verbose:
            print(f"\nMCMC Warm-up ({warmup_mcmc} steps)")
        accepted_swaps = 0
        # Run warm-up loop using tqdm (progress bar) for visualization
        self._initialize_likelihood_cache(self.warmup_non_edges)
        for _ in trange(warmup_mcmc, desc="MCMC Warm-up"):
            accepted = self._mcmc_step_for_permutation()
            if accepted:
                accepted_swaps += 1
        if verbose:
            print(f"Warm-up acceptance rate: {accepted_swaps/(warmup_mcmc or 1):.2%}")

        # --- Phase 2: Main optimization loop (gradient on P + MCMC permutation updates)
        if verbose:
            print(f"\nMain Optimization ({iterations} iterations)")
        best_ll = float('-inf')

        for iteration in range(iterations):
            iter_start_time = time.time()

            self.optimizer.zero_grad()
            
            # Use precomputed non-edges from training pool
            start = self.non_edge_ptr
            end = start + grad_samples
            if end > len(self.train_non_edges):
                start, end = 0, grad_samples  # wrap around
            self.non_edge_ptr = end
            non_edge_sample = self.train_non_edges[start:end]

            self._initialize_likelihood_cache(non_edge_sample)
            loss = -self.current_ll
            # Optimizers in PyTorch minimize by default, so we negate it
            loss.backward() # backprop: compute gradients of loss wrt P
            torch.nn.utils.clip_grad_norm_([self.P], max_norm=1.0)
            self.optimizer.step() # update P using Adam optimizer

            with torch.no_grad():
                self.P.clamp_(1e-6, 0.99999)

            mcmc_accepted = 0
            for _ in range(mcmc_per_iter):
                accepted = self._mcmc_step_for_permutation()
                if accepted:
                    mcmc_accepted += 1

            current_ll = -loss.item()
            if current_ll > best_ll:
                best_ll = current_ll
                best_theta = self.P.detach().cpu().numpy().flatten().tolist()

            if verbose:
                print(f"\n{iteration+1:3d}/{iterations}] LL: {current_ll:9.2f} , "
                    f"Best LL: {best_ll:9.2f}, MCMC Rate: {mcmc_accepted/(mcmc_per_iter or 1):5.1%} , "
                    f"Time: {(time.time() - iter_start_time):.1f}s")
                print(f"  P = {self.P[0,0].item():.4f}, {self.P[0,1].item():.4f}")
                print(f"      {self.P[1,0].item():.4f}, {self.P[1,1].item():.4f}")
            
        total_time = time.time() - start_time
        return {
            "theta": best_theta,
            "best_ll": best_ll,
            "time": total_time
        }
