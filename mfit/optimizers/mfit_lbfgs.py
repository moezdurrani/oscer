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
        self.inverse_perm = self._calculate_inverse_perm()

        # LBFGS optimizer
        # max_iter: how many internal line-search steps per optimizer.step() call.
        #   Higher = more thorough per iteration but more closure evaluations.
        #   20 is a good default for 4 parameters.
        # history_size: how many past gradients to store for curvature approximation.
        #   Since P only has 4 elements, a small history (10) is sufficient.
        # line_search_fn: 'strong_wolfe' enables a proper line search which is much
        #   more stable than the default (no line search). Strongly recommended.
        self.optimizer = torch.optim.LBFGS(
            [self.P],
            lr=1.0,                        # LBFGS uses line search, so lr=1.0 is standard
            max_iter=20,
            history_size=10,
            line_search_fn='strong_wolfe'
        )

        # Store the current non-edge sample so the closure can access it
        self._current_non_edge_sample = None

        self.warmup_mcmc = warmup_mcmc
        self.grad_samples = grad_samples
        self.iterations = iterations
        total_needed = self.warmup_mcmc + (self.iterations * grad_samples)
        total_needed = int(total_needed * 1.1)

        print(f"\n[Precomputing {total_needed:,} non-edges for training]")
        self.all_non_edges = self._sample_non_edges(total_needed)

        self.warmup_non_edges = self.all_non_edges[:warmup_mcmc]
        self.train_non_edges = self.all_non_edges[warmup_mcmc:]
        self.non_edge_ptr = 0

        self.edge_ll = None
        self.non_edge_ll = None
        self.sampled_non_edges = None

    def _initialize_likelihood_cache(self, non_edge_sample):
        perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
        perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]
        probs_e = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
        self.edge_ll = torch.log(probs_e.clamp(min=1e-10))

        perm_u = self.inverse_perm[non_edge_sample[:, 0]]
        perm_v = self.inverse_perm[non_edge_sample[:, 1]]
        probs_ne = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
        scale = self.num_non_edges / len(non_edge_sample)
        self.non_edge_ll = torch.log((1 - probs_ne).clamp(min=1e-10)) * scale

        self.current_ll = self.edge_ll.sum() + self.non_edge_ll.sum()
        self.sampled_non_edges = non_edge_sample.clone()

    def _delta_ll_for_swap(self, u, v):
        affected_edge_idx = list(set(self.node_to_edges[u] + self.node_to_edges[v]))
        affected_edges = self.edge_list_tensor[affected_edge_idx]

        delta = -self.edge_ll[affected_edge_idx].sum()

        perm_u = self.inverse_perm[affected_edges[:, 0]]
        perm_v = self.inverse_perm[affected_edges[:, 1]]
        new_probs = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
        new_ll = torch.log(new_probs.clamp(min=1e-10))

        delta += new_ll.sum()
        self.edge_ll[affected_edge_idx] = new_ll

        return delta

    def _get_scaled_initial_matrix(self, init_matrix):
        p_np = np.array(init_matrix, dtype=np.float64)
        if self.n_edges > 0:
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
        k_range = torch.arange(self.k - 1, -1, -1, device=self.device)
        u_bits = (u_indices.unsqueeze(-1) >> k_range) & 1
        v_bits = (v_indices.unsqueeze(-1) >> k_range) & 1

        A = self.P[0, 0]
        B = self.P[0, 1]
        C = self.P[1, 0]
        D = self.P[1, 1]

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
        attempts = 0
        max_attempts = n_samples * 10
        while len(non_edges) < n_samples and attempts < max_attempts:
            u = random.randint(0, self.n_nodes - 1)
            v = random.randint(0, self.n_nodes - 1)
            if u == v:
                continue
            edge = (u, v) if self.is_directed else tuple(sorted((u, v)))
            if edge in self.edge_set:
                continue
            non_edges.append((u, v))
            attempts += 1
        return torch.tensor(non_edges, dtype=torch.int64, device=self.device)

    def _calculate_log_likelihood(self, edges_to_eval, non_edges_to_eval):
        permuted_edges_u = self.inverse_perm[edges_to_eval[:, 0]]
        permuted_edges_v = self.inverse_perm[edges_to_eval[:, 1]]
        prob_edges = self._get_kron_edge_prob_vectorized(permuted_edges_u, permuted_edges_v)
        ll_edges = torch.log(prob_edges.clamp(min=1e-10)).sum()

        if len(non_edges_to_eval) > 0:
            permuted_non_edges_u = self.inverse_perm[non_edges_to_eval[:, 0]]
            permuted_non_edges_v = self.inverse_perm[non_edges_to_eval[:, 1]]
            prob_non_edges = self._get_kron_edge_prob_vectorized(permuted_non_edges_u, permuted_non_edges_v)
            scale_factor = self.num_non_edges / len(non_edges_to_eval)
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

        delta = self._delta_ll_for_swap(pi, pj)

        if delta > 0 or torch.rand(1).to(self.device) < torch.exp(delta):
            self.current_ll += delta
            return True
        else:
            self.perm[i], self.perm[j] = pi, pj
            self.inverse_perm[pi], self.inverse_perm[pj] = i, j
            return False

    def fit(self, iterations, grad_samples, warmup_mcmc, mcmc_per_iter):
        print(f"\nMCMC Warm-up ({warmup_mcmc} steps)")
        accepted_swaps = 0
        self._initialize_likelihood_cache(self.warmup_non_edges)
        for _ in trange(warmup_mcmc, desc="MCMC Warm-up"):
            accepted = self._mcmc_step_for_permutation()
            if accepted:
                accepted_swaps += 1
        print(f"Warm-up acceptance rate: {accepted_swaps/(warmup_mcmc or 1):.2%}")

        print(f"\nMain Optimization ({iterations} iterations) — LBFGS")
        best_ll = float('-inf')

        for iteration in range(iterations):
            iter_start_time = time.time()

            # Pin the non-edge sample for this iteration so the closure uses
            # the same sample across all internal LBFGS line-search steps.
            start = self.non_edge_ptr
            end = start + grad_samples
            if end > len(self.train_non_edges):
                start, end = 0, grad_samples
            self.non_edge_ptr = end
            self._current_non_edge_sample = self.train_non_edges[start:end]

            # LBFGS requires a closure that:
            #   1. Clears gradients
            #   2. Recomputes the loss
            #   3. Calls loss.backward()
            #   4. Returns the loss value
            # It may be called multiple times per optimizer.step() for line search.
            def closure():
                self.optimizer.zero_grad()
                self._initialize_likelihood_cache(self._current_non_edge_sample)
                loss = -self.current_ll
                loss.backward()
                # Note: grad clipping inside closure is valid for LBFGS
                torch.nn.utils.clip_grad_norm_([self.P], max_norm=1.0)
                return loss

            loss = self.optimizer.step(closure)

            with torch.no_grad():
                self.P.clamp_(1e-6, 0.99999)

            mcmc_accepted = 0
            for _ in range(mcmc_per_iter):
                accepted = self._mcmc_step_for_permutation()
                if accepted:
                    mcmc_accepted += 1

            current_ll = -loss.item()
            best_ll = max(best_ll, current_ll)

            print(f"\n{iteration+1:3d}/{iterations}] LL: {current_ll:9.2f} , "
                  f"Best LL: {best_ll:9.2f}, MCMC Rate: {mcmc_accepted/(mcmc_per_iter or 1):5.1%} , "
                  f"Time: {(time.time() - iter_start_time):.1f}s")
            print(f"  P = {self.P[0,0].item():.4f}, {self.P[0,1].item():.4f}")
            print(f"      {self.P[1,0].item():.4f}, {self.P[1,1].item():.4f}")