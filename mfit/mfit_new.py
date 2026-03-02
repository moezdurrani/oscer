import networkx as nx
import numpy as np

class mfit:
    def __init__(self, data_path, init_matrix, learning_rate, warmup_mcmc, grad_samples, iterations):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Store parameters
        self.init_matrix = np.array(init_matrix).reshape(2, 2)
        self.learning_rate = learning_rate
        self.warmup_mcmc = warmup_mcmc
        self.grad_samples = grad_samples
        self.iterations = iterations
        self.n_nodes = None
        self.n_edges = None
        self.k = None
        self.is_directed = False
        self.edge_list_tensor = None

        # Convert graph to tensors
        self._prepare_graph(data_path)

    def _prepare_graph(self, data_path):
        graph = nx.read_edgelist(data_path, create_using=nx.DiGraph(), nodetype=int)
        self.is_directed = any(not graph.has_edge(v, u) for u, v in graph.edges())

        if not self.is_directed:
            print(f"Graph Mode: {'Directed' if self.is_directed else 'Undirected'} (Auto-Detected)")
    
        # remapping labels to clean number sequence
        self.graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="default")
        self.n_nodes = graph.number_of_nodes()
        self.k = int(np.ceil(np.log2(self.n_nodes)))

        padded_n_nodes = 2 ** self.k
        if padded_n_nodes != self.n_nodes:
            print(f"Padding graph from {self.n_nodes} nodes to {padded_n_nodes} for model")
            graph.add_nodes_from(range(self.n_nodes, padded_n_nodes))
            self.n_nodes = padded_n_nodes
        self.n_edges = graph.number_of_edges()

        self.edge_list_tensor = torch.tensor(list(graph.edges()), dtype=torch.long, device=self.device)



    