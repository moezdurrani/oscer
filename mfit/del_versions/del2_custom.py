import torch
import numpy as np
import random
import time
import networkx as nx
from tqdm import trange

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

        graph_raw = graph_temp if self.is_directed else graph_temp.to_undirected()
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

        # Plain float64 tensor — no autograd, matches kronfit
        self.P = torch.tensor(scaled_p_np, dtype=torch.float64, device=self.device)
        self.lr = learning_rate

        self.perm = self._initialize_permutation_by_degree()
        self.inverse_perm = self._calculate_inverse_perm()

        self.warmup_mcmc = warmup_mcmc
        self.grad_samples = grad_samples
        self.iterations = iterations

        # We only need non-edges for the MCMC cache (not for gradient)
        # Precompute enough for warmup + main loop MCMC
        total_needed = int((warmup_mcmc + iterations * grad_samples) * 1.1)
        print(f"\n[Precomputing {total_needed:,} non-edges for MCMC cache]")
        self.all_non_edges = self._sample_non_edges(total_needed)
        self.warmup_non_edges = self.all_non_edges[:warmup_mcmc]
        self.train_non_edges = self.all_non_edges[warmup_mcmc:]
        self.non_edge_ptr = 0

        self.edge_ll = None
        self.non_edge_ll = None
        self.sampled_non_edges = None
        self.current_ll = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _get_scaled_initial_matrix(self, init_matrix):
        """Scale matrix so sum = E^(1/k), matching kronfit's SetForEdges."""
        p_np = np.array(init_matrix, dtype=np.float64)
        if self.n_edges > 0 and self.k > 0:
            current_sum = np.sum(p_np)
            target_sum = self.n_edges ** (1.0 / self.k)
            scale = target_sum / current_sum
            p_np = np.clip(p_np * scale, 1e-6, 0.999)
        return p_np

    def _initialize_permutation_by_degree(self):
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        return torch.tensor(sorted_nodes, dtype=torch.long, device=self.device)

    def _calculate_inverse_perm(self):
        inverse_perm = torch.empty_like(self.perm)
        inverse_perm[self.perm] = torch.arange(self.n_nodes, device=self.device)
        return inverse_perm

    # ------------------------------------------------------------------
    # Core probability computation
    # ------------------------------------------------------------------

    def _get_kron_edge_prob_vectorized(self, u_indices, v_indices):
        k_range = torch.arange(self.k - 1, -1, -1, device=self.device)
        u_bits = ((u_indices.unsqueeze(-1) >> k_range) & 1).to(torch.float64)
        v_bits = ((v_indices.unsqueeze(-1) >> k_range) & 1).to(torch.float64)
        A, B, C, D = self.P[0,0], self.P[0,1], self.P[1,0], self.P[1,1]
        probs_k = ((1-u_bits)*(1-v_bits)*A + (1-u_bits)*v_bits*B +
                   u_bits*(1-v_bits)*C + u_bits*v_bits*D)
        return torch.prod(probs_k, dim=1)

    def _get_bit_counts(self, u_indices, v_indices):
        """For each edge, count how many of the k bit-pairs fall into AA, AB, BA, BB."""
        k_range = torch.arange(self.k - 1, -1, -1, device=self.device)
        u_bits = ((u_indices.unsqueeze(-1) >> k_range) & 1).to(torch.float64)
        v_bits = ((v_indices.unsqueeze(-1) >> k_range) & 1).to(torch.float64)
        cnt_AA = ((1-u_bits)*(1-v_bits)).sum(dim=1)
        cnt_AB = ((1-u_bits)*v_bits).sum(dim=1)
        cnt_BA = (u_bits*(1-v_bits)).sum(dim=1)
        cnt_BB = (u_bits*v_bits).sum(dim=1)
        return cnt_AA, cnt_AB, cnt_BA, cnt_BB

    # ------------------------------------------------------------------
    # Gradient — exact match to kronfit's CalcApxGraphDLL
    #
    # Kronfit uses a 2nd-order Taylor approximation for the non-edge term:
    #   log(1-x) ≈ -x - 0.5*x^2
    #
    # This gives an O(E + N0) gradient (no non-edge sampling needed):
    #
    #   GetApxEmptyGraphDLL(param) =
    #       -k * Sum^(k-1) - k * SumSq^(k-1) * P[param]
    #   where Sum = sum of all P[i], SumSq = sum of P[i]^2
    #
    # Then for each edge (u,v):
    #   DLL += GetEdgeDLL(param, u, v)       = cnt_ab / P[a,b]
    #   DLL -= GetApxNoEdgeDLL(param, u, v)  = -cnt*exp(DLL_path) - cnt*exp(P[a,b] + 2*DLL_path)
    #
    # This is what kronfit's SampleGradient averages over NSamples MCMC steps.
    # We compute it once per iteration (equivalent to NSamples=1 avg).
    # ------------------------------------------------------------------

    def _compute_apx_empty_graph_dll(self):
        """
        GetApxEmptyGraphDLL for all 4 parameters at once.
        Vectorized version of:
            -KronIters * pow(Sum, KronIters-1)
            -KronIters * pow(SumSq, KronIters-1) * P[param]
        """
        P_flat = self.P.flatten()  # [A, B, C, D]
        Sum = P_flat.sum()
        SumSq = (P_flat ** 2).sum()
        k = self.k

        # Both terms, shape [4] -> reshape to [2,2]
        term1 = -k * (Sum ** (k - 1))
        term2 = -k * (SumSq ** (k - 1)) * P_flat
        dll_flat = term1 + term2
        return dll_flat.reshape(2, 2)

    def _compute_apx_no_edge_dll_vectorized(self, u_perm, v_perm):
        """
        Vectorized GetApxNoEdgeDLL for all 4 params and all edges at once.

        For param (a,b) and edge (u,v):
            cnt = number of bit-pairs equal to (a,b)  [= cnt_ab]
            DLL_path = sum of log(P[bit_r, bit_c]) for all k levels EXCEPT
                       one instance of (a,b) (i.e. sum - log(P[a,b]))
                     = log(edge_prob) - cnt * log(P[a,b])   ... but for cnt>1 it's trickier

        Actually the C++ formula:
            DLL = sum of LLMtx[X,Y] for all levels, skipping the FIRST occurrence of (a,b)
            ThetaCnt = count of (a,b) occurrences
            return -ThetaCnt * exp(DLL) - ThetaCnt * exp(P[a,b] + 2*DLL)

        Where LLMtx[i,j] = log(P[i,j]).

        DLL (the path sum) = log(edge_prob) - log(P[a,b])
            because we sum all k levels but skip one (a,b).

        So: path_sum = log(P_edge) - log(P[a,b])
            apx_no_edge_dll[a,b] = -cnt_ab * exp(path_sum)
                                   -cnt_ab * exp(log(P[a,b]) + 2*path_sum)
        """
        with torch.no_grad():
            # log edge probability for each edge: sum of log(P[bit,bit]) over k levels
            k_range = torch.arange(self.k - 1, -1, -1, device=self.device)
            u_bits = ((u_perm.unsqueeze(-1) >> k_range) & 1).to(torch.float64)
            v_bits = ((v_perm.unsqueeze(-1) >> k_range) & 1).to(torch.float64)

            log_P = torch.log(self.P.clamp(min=1e-10))
            A, B, C, D = log_P[0,0], log_P[0,1], log_P[1,0], log_P[1,1]

            # log prob contribution at each level
            log_probs_k = ((1-u_bits)*(1-v_bits)*A + (1-u_bits)*v_bits*B +
                           u_bits*(1-v_bits)*C + u_bits*v_bits*D)
            log_edge_prob = log_probs_k.sum(dim=1)  # [n_edges]

            cnt_AA, cnt_AB, cnt_BA, cnt_BB = self._get_bit_counts(u_perm, v_perm)
            cnts = [cnt_AA, cnt_AB, cnt_BA, cnt_BB]

            grad = torch.zeros(2, 2, dtype=torch.float64, device=self.device)
            params = [(0,0), (0,1), (1,0), (1,1)]

            for (i, j), cnt in zip(params, cnts):
                log_p_ab = log_P[i, j]
                # path_sum = log(edge_prob) - log(P[a,b])
                # (remove one instance of (a,b) from the product)
                path_sum = log_edge_prob - log_p_ab  # [n_edges]
                apx_dll = -cnt * torch.exp(path_sum) - cnt * torch.exp(log_p_ab + 2 * path_sum)
                grad[i, j] = apx_dll.sum()

            return grad

    def _compute_gradient(self):
        """
        Exact GPU replica of kronfit's CalcApxGraphDLL:

            GradV[param] = GetApxEmptyGraphDLL(param)
            for each edge (u,v):
                GradV[param] += GetEdgeDLL(param, u, v)
                GradV[param] -= GetApxNoEdgeDLL(param, u, v)

        No non-edge sampling. O(E + N0). Pure GPU.
        """
        with torch.no_grad():
            # Empty graph approximate gradient term (O(N0))
            grad = self._compute_apx_empty_graph_dll()

            # Edge terms
            perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
            perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]

            cnt_AA, cnt_AB, cnt_BA, cnt_BB = self._get_bit_counts(perm_u, perm_v)

            # GetEdgeDLL: cnt_ab / P[a,b]
            grad[0,0] += (cnt_AA / self.P[0,0].clamp(min=1e-10)).sum()
            grad[0,1] += (cnt_AB / self.P[0,1].clamp(min=1e-10)).sum()
            grad[1,0] += (cnt_BA / self.P[1,0].clamp(min=1e-10)).sum()
            grad[1,1] += (cnt_BB / self.P[1,1].clamp(min=1e-10)).sum()

            # GetApxNoEdgeDLL: subtract the approx non-edge contribution per edge
            grad -= self._compute_apx_no_edge_dll_vectorized(perm_u, perm_v)

            return grad

    # ------------------------------------------------------------------
    # Likelihood for reporting (uses sampled non-edges for efficiency)
    # ------------------------------------------------------------------

    def _compute_ll(self, non_edge_sample):
        with torch.no_grad():
            perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
            perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]
            probs_e = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
            ll_edges = torch.log(probs_e.clamp(min=1e-10)).sum()

            perm_u_ne = self.inverse_perm[non_edge_sample[:, 0]]
            perm_v_ne = self.inverse_perm[non_edge_sample[:, 1]]
            probs_ne = self._get_kron_edge_prob_vectorized(perm_u_ne, perm_v_ne)
            scale = self.num_non_edges / len(non_edge_sample)
            ll_non_edges = torch.log((1 - probs_ne).clamp(min=1e-10)).sum() * scale

            return (ll_edges + ll_non_edges).item()

    # ------------------------------------------------------------------
    # Non-edge sampling (used only for MCMC cache, not gradient)
    # ------------------------------------------------------------------

    def _sample_non_edges(self, n_samples):
        if USE_FAST_SAMPLING:
            if self.is_directed:
                non_edges_np = sample_non_edges_directed_numba(
                    self.n_nodes, self.edge_array_for_numba, n_samples)
            else:
                non_edges_np = sample_non_edges_undirected_numba(
                    self.n_nodes, self.edge_array_for_numba, n_samples)
            return torch.tensor(non_edges_np, dtype=torch.long, device=self.device)

        non_edges = []
        attempts = 0
        max_attempts = n_samples * 20
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

    # ------------------------------------------------------------------
    # MCMC permutation (Metropolis-Hastings on node permutation)
    # ------------------------------------------------------------------

    def _initialize_likelihood_cache(self, non_edge_sample):
        with torch.no_grad():
            perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
            perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]
            probs_e = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
            self.edge_ll = torch.log(probs_e.clamp(min=1e-10))

            perm_u_ne = self.inverse_perm[non_edge_sample[:, 0]]
            perm_v_ne = self.inverse_perm[non_edge_sample[:, 1]]
            probs_ne = self._get_kron_edge_prob_vectorized(perm_u_ne, perm_v_ne)
            self.non_edge_ll = torch.log((1 - probs_ne).clamp(min=1e-10))
            self.sampled_non_edges = non_edge_sample

            scale = self.num_non_edges / len(non_edge_sample)
            self.current_ll = (self.edge_ll.sum() + self.non_edge_ll.sum() * scale).item()

    def _delta_ll_for_swap(self, pi, pj):
        with torch.no_grad():
            # Edges affected by nodes pi or pj
            affected_e_idx = torch.tensor(
                self.node_to_edges.get(pi, []) + self.node_to_edges.get(pj, []),
                dtype=torch.long, device=self.device)

            edge_delta = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            new_edge_ll = None
            if len(affected_e_idx) > 0:
                affected_edges = self.edge_list_tensor[affected_e_idx]
                perm_u = self.inverse_perm[affected_edges[:, 0]]
                perm_v = self.inverse_perm[affected_edges[:, 1]]
                new_probs = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
                new_edge_ll = torch.log(new_probs.clamp(min=1e-10))
                edge_delta = new_edge_ll.sum() - self.edge_ll[affected_e_idx].sum()

            # Non-edges affected by nodes pi or pj
            ne = self.sampled_non_edges
            mask = ((ne[:, 0] == pi) | (ne[:, 1] == pi) |
                    (ne[:, 0] == pj) | (ne[:, 1] == pj))
            affected_ne_idx = mask.nonzero(as_tuple=True)[0]

            ne_delta = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            new_ne_ll = None
            if len(affected_ne_idx) > 0:
                affected_ne = ne[affected_ne_idx]
                perm_u_ne = self.inverse_perm[affected_ne[:, 0]]
                perm_v_ne = self.inverse_perm[affected_ne[:, 1]]
                new_probs_ne = self._get_kron_edge_prob_vectorized(perm_u_ne, perm_v_ne)
                new_ne_ll = torch.log((1 - new_probs_ne).clamp(min=1e-10))
                scale = self.num_non_edges / len(ne)
                ne_delta = (new_ne_ll.sum() - self.non_edge_ll[affected_ne_idx].sum()) * scale

            return (edge_delta + ne_delta), affected_e_idx, new_edge_ll, affected_ne_idx, new_ne_ll

    def _mcmc_step_for_permutation(self):
        i = torch.randint(0, self.n_nodes, (1,)).item()
        j = torch.randint(0, self.n_nodes, (1,)).item()
        if i == j:
            return False

        pi, pj = self.perm[i].item(), self.perm[j].item()

        # Apply swap
        self.perm[i], self.perm[j] = self.perm[j].clone(), self.perm[i].clone()
        self.inverse_perm[pi], self.inverse_perm[pj] = j, i

        delta, affected_e_idx, new_edge_ll, affected_ne_idx, new_ne_ll = \
            self._delta_ll_for_swap(pi, pj)

        if delta.item() > 0 or torch.rand(1).item() < torch.exp(torch.clamp(delta, max=0)).item():
            # Accept: commit cache
            if len(affected_e_idx) > 0:
                self.edge_ll[affected_e_idx] = new_edge_ll
            if len(affected_ne_idx) > 0:
                self.non_edge_ll[affected_ne_idx] = new_ne_ll
            self.current_ll += delta.item()
            return True
        else:
            # Reject: revert swap
            self.perm[i], self.perm[j] = self.perm[j].clone(), self.perm[i].clone()
            self.inverse_perm[pi], self.inverse_perm[pj] = i, j
            return False

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self, mcmc_per_iter=1000, verbose=True):
        grad_samples = self.grad_samples
        iterations = self.iterations

        # MCMC warm-up
        if verbose:
            print(f"\nMCMC Warm-up ({self.warmup_mcmc} steps)")
        self._initialize_likelihood_cache(self.warmup_non_edges)
        warmup_accepted = 0
        for _ in trange(self.warmup_mcmc, desc="MCMC Warm-up", disable=not verbose):
            if self._mcmc_step_for_permutation():
                warmup_accepted += 1
        if verbose:
            print(f"Warm-up acceptance rate: {100*warmup_accepted/self.warmup_mcmc:.2f}%")
            perm_u = self.inverse_perm[self.edge_list_tensor[:, 0]]
            perm_v = self.inverse_perm[self.edge_list_tensor[:, 1]]
            probs_e = self._get_kron_edge_prob_vectorized(perm_u, perm_v)
            print(f"Post-warmup edge probs: min={probs_e.min():.6f}, mean={probs_e.mean():.6f}, max={probs_e.max():.6f}")
            print(f"Post-warmup current_ll: {self.current_ll:.2f}")

        if verbose:
            print(f"\nMain Optimization ({iterations} iterations)")

        best_ll = float('-inf')
        best_theta = None

        # Per-parameter adaptive learning rates, matching kronfit's LearnRateV
        learn_rate = torch.full((2, 2), self.lr, dtype=torch.float64, device=self.device)

        MxStep = 0.05
        MnStep = 0.005

        for iteration in range(iterations):
            iter_start_time = time.time()

            start = self.non_edge_ptr
            end = start + grad_samples
            if end > len(self.train_non_edges):
                start, end = 0, grad_samples
            self.non_edge_ptr = end
            non_edge_sample = self.train_non_edges[start:end]

            self._initialize_likelihood_cache(non_edge_sample)

            # SampleGradient: compute initial gradient, then accumulate over accepted MCMC swaps
            avg_grad = self._compute_gradient()
            avg_ll = self.current_ll
            mcmc_accepted = 0
            for _ in range(mcmc_per_iter):
                if self._mcmc_step_for_permutation():
                    mcmc_accepted += 1
                    avg_grad = avg_grad + self._compute_gradient()
                avg_ll += self.current_ll
            n_grad_samples = 1 + mcmc_accepted
            avg_grad = avg_grad / n_grad_samples
            avg_ll = avg_ll / (1 + mcmc_per_iter)
            grad = avg_grad

            # Adaptive learning rate (kronfit style)
            learn_rate *= 0.95
            with torch.no_grad():
                new_P = self.P.clone()
                for i in range(2):
                    for j in range(2):
                        g = grad[i, j].item()
                        lr = learn_rate[i, j].item()
                        if iteration < 1:
                            while abs(lr * g) > MxStep: lr *= 0.95
                            while abs(lr * g) < 0.02:   lr *= (1.0 / 0.95)
                        else:
                            while abs(lr * g) > MxStep: lr *= 0.95
                            while abs(lr * g) < MnStep:  lr *= (1.0 / 0.95)
                        learn_rate[i, j] = lr
                        new_P[i, j] = self.P[i, j] + lr * g
                        new_P[i, j] = new_P[i, j].clamp(0.0001, 0.9999)
                self.P = new_P
                if MxStep > 3 * MnStep:
                    MxStep *= 0.95

            current_ll = self._compute_ll(non_edge_sample)
            if current_ll > best_ll:
                best_ll = current_ll
                best_theta = self.P.cpu().numpy().flatten().tolist()

            if verbose:
                elapsed = time.time() - iter_start_time
                mcmc_rate = 100 * mcmc_accepted / mcmc_per_iter
                print(f"\n[{iteration+1:3d}/{iterations}] LL: {current_ll:10.2f} , "
                      f"Best LL: {best_ll:10.2f}, MCMC Rate: {mcmc_rate:.1f}% , "
                      f"Time: {elapsed:.1f}s")
                p = self.P.cpu().numpy()
                print(f"  P    = {p[0,0]:.4f}, {p[0,1]:.4f}")
                print(f"         {p[1,0]:.4f}, {p[1,1]:.4f}")
                g = grad.cpu().numpy()
                print(f"  Grad = {g[0,0]:.6f}, {g[0,1]:.6f}")
                print(f"         {g[1,0]:.6f}, {g[1,1]:.6f}")

        return {
            "theta": best_theta,
            "best_ll": best_ll,
            "final_P": self.P.cpu().numpy().tolist()
        }